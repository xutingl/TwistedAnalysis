def _ragged_a2a_kernel_point_to_point(
    input_offsets_ref: Ref,
    output_offsets_ref: Ref,
    sizes_ref: Ref,
    total_send_amount_ref: Ref,
    total_recv_amount_ref: Ref,
    num_packets_per_group_ref: Ref,
    x_ref: Ref,
    _: Ref,
    o_ref: Ref,
    scratch_ref: Ref | None,
    send_sem: Ref,
    recv_sem: Ref,
    scratch_sems: Ref | None,
    *,
    axis_name: str | tuple[str, ...],
    transpose: bool,
    packet_size: int,
    enable_checks: bool = False,
):
  """Kernel for ragged all-to-all.
 
  Args:
    input_offsets_ref: An int32[m] Ref where each entry is the start index for
      each group in the all to all.
    output_offsets_ref: An int32[m] Ref where each entry is the destination
      index for each group in the all to all.
    sizes_ref: An int32[m] Ref where each entry is the amount of each group to
      send.
    total_send_amount_ref: A int32[1] Ref where the entry is the total amount of
      data to send.
    total_recv_amount_ref: A int32[1] Ref where the entry is the total amount of
      data to receive.
    num_packets_per_group_ref: The number of packets to send.
    x_ref: The input Ref.
    _: Alias to o_ref.
    o_ref: The destination Ref.
    scratch_ref: An unused scratch space Ref.
    send_sem: A semaphore used to track send DMAs.
    recv_sem: A semaphore used to track recv DMAs.
    scratch_sems: Unused semaphores to sync scratch_ref DMAs.
    axis_name: The name of the axis to perform the all-to-all on.
    transpose: Whether or not this all-to-all is transposed.
    packet_size: The packet size to use for the DMA.
    enable_checks: Whether or not to enable ``pl.debug_check``s of rare
      circumstances.
  """
  assert scratch_ref is None
  del scratch_ref
  assert scratch_sems is None
  del scratch_sems
  # TODO(later): we need to call axis_index because pallas_call won't think
  # it's a collective otherwise. Fix this.
  my_id = jax.lax.axis_index(axis_name)
  axis_size = jax.lax.axis_size(axis_name)
  num_groups = sizes_ref.shape[0]
 
  if axis_size > 1:
    ragged_collectives_utils.main_barrier(
        axis_name, barrier_type=ragged_collectives_utils.BarrierType.ALL_TO_ALL
    )
 
  groups_per_shard, r = divmod(num_groups, axis_size)
  assert r == 0, (num_groups, axis_size)
  expert_offset = (my_id + 1) * groups_per_shard
 
  def inner_body(i, _):
    # We iterate across the groups first, so that we schedule packets to
    # different devices in a round-robin fashion.
    packet_idx = lax.div(i, num_groups)
    # On device_0, we want to start sending data of the first group on device_1.
    # On device_1, we want to start sending data of the first group on device_2,
    # etc.
    group_idx = lax.rem(i + expert_offset, num_groups)
    if transpose:
      # In a transposed all-to-all, we are sending data back to its origin.
      # Since we received data from each other device contiguously, our data is
      # layed out like `[g_0,..., g_d, g_0,..., g_d]` where d is the number of
      # devices. So when we loop over our groups, the target device is
      # group_idx % d.
      target_device = lax.rem(group_idx, axis_size)
    else:
      # In a regular all-to-all, we are sending data laid out like this:
      # `[g_0,...,g_m]` where we have m groups but d devices. Therefore there
      # are m / d groups per device next to each other in our data. To compute
      # which device to send each group to, we therefore need to compute
      # group_idx / (m / d).
      target_device = lax.div(group_idx, groups_per_shard)
 
    size = lax.min(
        packet_size, lax.max(sizes_ref[group_idx] - packet_idx * packet_size, 0)
    )
 
    if enable_checks:
      pl.debug_check(sizes_ref[group_idx] >= 0, "Found group size < 0.")
      pl.debug_check(size >= 0, "Found transfer size < 0.")
 
    # Avoid doing zero-sized remote DMAs
    @pl.when(size > 0)
    def _():
      input_offset = input_offsets_ref[group_idx] + packet_idx * packet_size
      output_offset = output_offsets_ref[group_idx] + packet_idx * packet_size
      if enable_checks:
        pl.debug_check(input_offset >= 0, "Found input_offset < 0.")
        pl.debug_check(output_offset >= 0, "Found output_offset < 0.")
        pl.debug_check(
            input_offset + size <= x_ref.shape[0],
            "Found input_offset + size > x_ref.shape[0].",
        )
        pl.debug_check(
            output_offset + size <= o_ref.shape[0],
            "Found output_offset + size > o_ref.shape[0].",
        )
      if axis_size > 1:
        pltpu.make_async_remote_copy(
            x_ref.at[pl.ds(input_offset, size)],
            o_ref.at[pl.ds(output_offset, size)],
            device_id={axis_name: target_device},
            send_sem=send_sem,
            recv_sem=recv_sem,
        ).start()
      else:
        pltpu.make_async_copy(
            x_ref.at[pl.ds(input_offset, size)],
            o_ref.at[pl.ds(output_offset, size)],
            sem=send_sem,
        ).start()
 
  # We do num_groups * num_packets_per_group_ref[0] instead of x.shape[0]
  # because this better accounts for buffer with lots of padding.
  jax.lax.fori_loop(
      0, num_groups * num_packets_per_group_ref[0], inner_body, None
  )
 
  # We've sent out all the data, now we need to wait for data to arrive from
  # other chips. Since the amount each chip sends to the other is dynamic,
  # we will likely receive different amounts of data than we sent.
  send_amount = total_send_amount_ref[0]
  recv_amount = total_recv_amount_ref[0]
  if enable_checks:
    pl.debug_check(send_amount >= 0, "Found send_amount < 0.")
    pl.debug_check(recv_amount >= 0, "Found recv_amount < 0.")
    pl.debug_check(
        send_amount <= x_ref.shape[0], f"Found send_amount > {x_ref.shape=}."
    )
    pl.debug_check(
        recv_amount <= o_ref.shape[0], f"Found recv_amount > {o_ref.shape=}."
    )
 
  # To wait for a specific amount of data to be sent/received, we'll construct a
  # dummy DMA that correspond to that exact amount of bytes on the send/recv
  # semaphore and wait for it.
  pltpu.make_async_copy(
      o_ref.at[pl.ds(0, send_amount)], o_ref.at[pl.ds(0, send_amount)], send_sem
  ).wait()
  if axis_size > 1:
    pltpu.make_async_copy(
        o_ref.at[pl.ds(0, recv_amount)],
        o_ref.at[pl.ds(0, recv_amount)],
        recv_sem,
    ).wait()


@differentiable_ragged_all_to_all.add_ra2a_vjp
def ragged_all_to_all(
    x: jax.Array,
    routing_info: ragged_routing.RoutingInfo,
    existing_out: jax.Array | None = None,
    *,
    mesh: Mesh,
    output_buffer_size: int,
    axis_name: str | tuple[str, ...],
    collective_id: int,
    transpose: bool = False,
    mask_value: int | float | None = 0,
    kernel_impl: KernelImpl | None = None,
    promise_batch_alignment: int | None = None,
    fake_quant_x: bool = False,
    align_tokens_to: int = 1,
    pack_4_in_int32: bool = False,
    use_vmem_bounce_kernel: bool = False,
) -> jax.Array:
  """Performs a ragged all-to-all collective over a mesh axis.

  Args:
    x: The array to perform the all-to-all on. It should be a ragged array in a
      concatenated layout along the leading dimension, i.e. `[x_0; x_1; ..;
      x_m]`, where each of `x_i` can have a different length. The total length
      of x is the sum lengths of each of `x_i` and `m` is the total number of
      "groups" in this array.
    routing_info: A RoutingInfo object that contains the information necessary
      to compute input_offsets, output_offsets, send_sizes, and recv_sizes.
    existing_out: An optional array to store the output of the all to all.
    mesh: A JAX mesh.
    output_buffer_size: The size of the output buffer. This max number of
      elements a device can receive from the other devices in the all to all.
    axis_name: The axis name the collective will happen over.
    collective_id: A unique identifier for this collective.
    transpose: A boolean indicating whether or not this all-to-all is
      transposed. If transposed, rather than sending the `i`-th group to device
      `i // num_devices`, we send the `i`-th group to device `i % num_devices`.
      When transpose=True, this is a reverse of the forward operation.
    mask_value: A value that if provided is used to mask out garbage in the
      output. If existing_out is provided, this is unused.
    kernel_impl: The implementation of the kernel to use. None is advised, so
      that the kernel implementation is picked automatically.
    promise_batch_alignment: If provided, we are guaranteed that the number of
      tokens sent to each expert will be a multiple of this number.
    fake_quant_x: If True, we will apply emulated quant to x in preparation for
      landing eventual actual quant.
    align_tokens_to: Align the output offsets to this value.
    pack_4_in_int32: If True, packs 4 elements into an int32. Only used for
      multihop kernel.
    use_vmem_bounce_kernel: If True, use the VMEM bounce kernel for the local
      copy. (Only supported for SparseCore kernel.)

  Returns:
    An array of shape `[output_buffer_size, *x.shape[1:]]` with the tokens on
    each device shard according to the groups in the routing_info specification.
  """
  kernel_impl = kernel_impl or get_default_kernel_impl(mesh)
  assert (
      not use_vmem_bounce_kernel or kernel_impl == KernelImpl.SPARSECORE
  ), "use_vmem_bounce_kernel is not supported."

  # Apply emulated quant to x in preparation for landing eventual actual quant.
  if fake_quant_x:
    x: jax.Array = quantized_matmul.roundtrip_quant_dequant(x, axis=-1)
  if kernel_impl == KernelImpl.MULTIHOP:
    # In the case of the power 2 hop implementation, we have a different api
    # due to different metadata requirements - so we return directly from here.
    assert isinstance(
        mesh, jax.sharding.Mesh
    ), "Multihop kernel requires a concrete mesh"
    result, _ = ragged_all_to_all_vlp_multihop.ragged_all_to_all_multihop(
        x,
        routing_info,
        existing_out=existing_out,
        mesh=mesh,
        output_buffer_size=output_buffer_size,
        axis_name=axis_name,
        collective_id=collective_id,
        transpose=transpose,
        permissible_distances=None,
        num_cycles_override=None,
        mask_value=mask_value,
        pack_4_in_int32=pack_4_in_int32,
    )
    return result

  if x.ndim < 1:
    raise ValueError(f"x must be at least 1D, got {x.ndim=}")

  # Callbacks to undo the transformations made to the input, so the output comes
  # out in the same shape as the input.
  cleanups = []

  def cleanup_out(out):
    for f in reversed(cleanups):
      out = f(out)
    return out

  if x.ndim == 1:
    x = x[..., None]
    if existing_out is not None:
      existing_out = existing_out[..., None]
    cleanups.append(lambda out: out[..., 0])

  # Currently, we can only do dynamic slices along untiled dimensions, but our
  # input `x` is [b, d], meaning the ragged `b` dimension is tiled along
  # sublanes. We need to reshape out the `b` dimension into untiled. Ideally
  # we do something like [b, d] -> [b, 1, d], leading to a (1, 128) tiling.
  # For sub-4 byte dtypes, we need to pad to a multiple of packing * 128 bytes
  # and then reshape to [b, packing, -1]
  # Examples:
  # f32[128, 128] -> f32[128, 1, 128]
  # bf16[128, 128] -> bf16[128, 256] -> bf16[128, 2, 128]
  # f32[128, 257] -> f32[128, 384] -> f32[128, 1, 384]
  # bf16[128, 257] -> bf16[128, 512] -> bf16[128, 2, 256]

  # If promise_batch_alignment is provided, we will instead reshape to
  # [B/promise_batch_alignment, promise_batch_alignment, D] relying on the
  # assertion that the ragged batch sent to each expert from each source shard
  # is appropriately aligned. We will adjust the input/output offsets and sizes
  # accordingly.

  axis_size = lax.axis_size(axis_name)
  if axis_size == 1:
    kernel_impl = KernelImpl.POINT_TO_POINT
    use_vmem_bounce_kernel = False
  device_index = (
      lax.axis_index(axis_name)
      if kernel_impl != KernelImpl.NEIGHBOUR_HOP
      else None
  )

  input_offsets = routing_info.get_input_offsets(
      device_index=device_index,
      transpose=transpose,
      align_output_offsets=align_tokens_to,
  )
  output_offsets = routing_info.get_output_offsets(
      device_index=device_index,
      transpose=transpose,
      align_output_offsets=align_tokens_to,
  )
  send_sizes = routing_info.get_send_sizes(
      device_index=device_index, transpose=transpose
  )
  recv_sizes = routing_info.get_recv_sizes(
      device_index=device_index, transpose=transpose
  )

  # TPUs work in 32 bit words, so this is the packing needed to fill words
  # while keeping the same chunk tiling. SparseCore wants us to be chunk
  # aligned, not just sublane aligned.
  packing = (
      32 // x.dtype.itemsize
      if kernel_impl == KernelImpl.SPARSECORE
      else 4 // x.dtype.itemsize
  )
  assert packing > 0

  # Pad minormost dimension to be lane aligned.
  padding_alignment = 128 * (packing if x.ndim == 2 else 1)
  *_, dim = x.shape
  if dim % padding_alignment != 0:
    dim_padding = (dim // padding_alignment + 1) * padding_alignment - dim
    x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, dim_padding)])
    if existing_out is not None:
      existing_out = jnp.pad(
          existing_out, [(0, 0)] * (x.ndim - 1) + [(0, dim_padding)]
      )
    cleanups.append(lambda out: out[..., :-dim_padding])

  if x.ndim == 2:
    # Reshape to make the ragged dimension major.
    if promise_batch_alignment is not None:
      assert x.shape[0] % promise_batch_alignment == 0, (
          "X shape not aligned with promised batch alignment: ",
          x.shape,
      )
      x = x.reshape(
          (x.shape[0] // promise_batch_alignment, promise_batch_alignment, -1)
      )
      if existing_out is not None:
        existing_out = existing_out.reshape((
            existing_out.shape[0] // promise_batch_alignment,
            promise_batch_alignment,
            -1,
        ))
      input_offsets //= promise_batch_alignment
      output_offsets //= promise_batch_alignment
      send_sizes //= promise_batch_alignment
      recv_sizes //= promise_batch_alignment
      output_buffer_size //= promise_batch_alignment

      cleanups.append(lambda out: out.reshape((-1, out.shape[-1])))
    else:
      x = x.reshape((x.shape[0], packing, -1))
      if existing_out is not None:
        existing_out = existing_out.reshape(
            (existing_out.shape[0], packing, -1)
        )
      cleanups.append(lambda out: out.reshape((out.shape[0], -1)))

  if kernel_impl == KernelImpl.SPARSECORE:
    if existing_out is None:
      if mask_value is None:
        with xla_metadata.set_xla_metadata(_scheduling_group_id="noop"):
          existing_out = pl.empty(
              (output_buffer_size, *x.shape[1:]), dtype=x.dtype
          )
      else:
        existing_out = jnp.full(
            (output_buffer_size, *x.shape[1:]), mask_value, dtype=x.dtype
        )

    out = ragged_all_to_all_sparsecore.ragged_all_to_all_impl(
        x,
        existing_out,
        send_sizes,
        recv_sizes,
        input_offsets,
        output_offsets,
        axis_name=axis_name,
        transpose=transpose,
        collective_id=collective_id,
        use_vmem_bounce_kernel=use_vmem_bounce_kernel,
    )
    return cleanup_out(out)

  if kernel_impl == KernelImpl.LAX:
    out = _lax_ragged_all_to_all_kernel(
        x,
        input_offsets,
        output_offsets,
        send_sizes,
        recv_sizes,
        existing_out=existing_out,
        output_buffer_size=output_buffer_size,
        axis_name=axis_name,
        transpose=transpose,
        mask_value=mask_value,
    )
    return cleanup_out(out)

  _, *ds = x.shape

  out_shape = jax.ShapeDtypeStruct((output_buffer_size, *ds), x.dtype)
  if existing_out is not None:
    assert existing_out.shape == out_shape.shape
    assert existing_out.dtype == out_shape.dtype

  if kernel_impl == KernelImpl.POINT_TO_POINT:
    expected_metadata_rank = 1
    kernel = _ragged_a2a_kernel_point_to_point
    scratch_spec = None
    scratch_sem_spec = None
    scratch_shape = None
  elif kernel_impl == KernelImpl.NEIGHBOUR_HOP:
    expected_metadata_rank = 2
    kernel = ragged_all_to_all_neighbour_hop.ragged_a2a_kernel_neighbour_hop
    scratch_spec = pl.BlockSpec(memory_space=pl.ANY)
    # We need 2 buffers for left and right; and two buffers so that we can send
    # to the left while still receiving from the right.
    # Finally we need a buffer send, recv and capacity semaphore.
    scratch_sem_spec = (
        pltpu.SemaphoreType.DMA((2, 2, 2)),
        pltpu.SemaphoreType.REGULAR((2, 2)),
    )
    scratch_shape = jax.ShapeDtypeStruct((2, 2) + x.shape, x.dtype)
  else:
    raise ValueError(f"Unknown kernel impl: {kernel_impl}")

  for k, v in {
      "input_offsets": input_offsets,
      "output_offsets": output_offsets,
      "send_sizes": send_sizes,
      "recv_sizes": recv_sizes,
  }.items():
    if v.ndim != expected_metadata_rank:
      raise ValueError(
          f"{k} must be {expected_metadata_rank}D, got {k}.shape={v.shape}."
      )
  total_send_amount = send_sizes.sum(axis=-1, keepdims=True)
  total_recv_amount = recv_sizes.sum(axis=-1, keepdims=True)
  input_output_aliases = {}
  input_output_aliases[7] = 0
  if existing_out is None and mask_value is None:
    existing_out = pl.empty((output_buffer_size, *ds), dtype=x.dtype)
  elif existing_out is None:
    existing_out = jnp.full(
        (output_buffer_size, *ds), mask_value, dtype=x.dtype
    )
  name = f"ragged_a2a_{kernel_impl.value}"
  if transpose:
    name += "_transpose"

  compiler_params = dict()

  if axis_size > 1:
    compiler_params = dict(collective_id=collective_id)
  with jax.named_scope(name):
    # Work out optimal packet size and number of iterations necessary to send
    # all the data to each expert.
    n_bytes_per_element = np.prod(x.shape[1:]) * jnp.dtype(x.dtype).itemsize
    # Measured in elements.
    # 2 ** 15 is the optimal XLA packet size.
    optimal_packet_size = int(2**15) // n_bytes_per_element
    packet_size = np.clip(optimal_packet_size, 1, x.shape[0], dtype=np.int32)
    # This is the maximum number of iterations we'll need to do per group.
    num_packets_per_group = jnp.array(
        [pl.cdiv(jnp.max(send_sizes), packet_size)]
    ).astype(jnp.int32)
    # The actual amount of bytes transferred is
    # `total_send_amount * (x_shape[1] * ... * x_shape[-1]) * x_itemsize`.
    # We use x_shape[0] to approximate total_send_amount, which is
    # `prod(x_shape) * x_itemsize`.
    cost_estimate = pl.CostEstimate(
        flops=0,
        bytes_accessed=0,
        transcendentals=0,
        remote_bytes_transferred=int(np.prod(x.shape)) * x.dtype.itemsize,
    )
    out, unused_scratch = pl.pallas_call(
        functools.partial(
            kernel,
            axis_name=axis_name,
            transpose=transpose,
            packet_size=packet_size,
        ),
        out_shape=(out_shape, scratch_shape),
        # collective_id is used in assigning barrier semaphores to ops. Ops with
        # the same collective_id can be assigned the same barrier.
        # We use PrefetchScalarGridSpec here so that we can use `scratch_shapes`
        # to allocate semaphores for the send and recv DMAs.
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                scratch_sem_spec,
            ),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pltpu.SMEM),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY)
                if existing_out is not None
                else None,
            ],
            out_specs=[pl.BlockSpec(memory_space=pl.ANY), scratch_spec],
        ),
        input_output_aliases=input_output_aliases,
        cost_estimate=cost_estimate,
        compiler_params=pltpu.CompilerParams(**compiler_params),
    )(
        input_offsets,
        output_offsets,
        send_sizes,
        total_send_amount,
        total_recv_amount,
        num_packets_per_group,
        x,
        existing_out,
    )
    return cleanup_out(out)
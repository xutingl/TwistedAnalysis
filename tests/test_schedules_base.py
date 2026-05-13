from twisted_analysis.schedules.base import Injection, ScheduleResult
from twisted_analysis.model.flow import Flow


def test_injection_dataclass():
    f = Flow((0, 0), (0, 1), 1)
    inj = Injection(flow=f, start_step=0)
    assert inj.flow == f and inj.start_step == 0


def test_schedule_result_ratio():
    res = ScheduleResult(name="test", makespan=20, lower_bound=10)
    assert res.ratio == 2.0

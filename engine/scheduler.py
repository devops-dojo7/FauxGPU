"""Multi-tenant GPU scheduling simulation: a discrete-event simulation of
priority-based scheduling with preemption over a fixed GPU pool. Given a
known list of jobs (team, priority, GPU count, submit time, duration) and a
pool size, computes the exact timeline of who ran when — including
preemption pauses and resumes — in one deterministic pass. Pure Python, no
I/O, no framework dependency.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

_ARRIVAL = 0
_COMPLETION = 1


@dataclass(frozen=True)
class SchedJob:
    job_id: str
    team: str
    priority: int  # higher = more important
    gpu_count: int
    submit_time: float
    duration: float  # total run time if never preempted


@dataclass(frozen=True)
class GpuPool:
    gpu_id: str  # catalog id, for display/pricing only — not load-bearing to the algorithm
    total_count: int


@dataclass(frozen=True)
class TimelineSegment:
    start: float
    end: float


@dataclass(frozen=True)
class JobOutcome:
    job_id: str
    team: str
    segments: tuple[TimelineSegment, ...]  # ordered run segments; gaps between them are preemption pauses
    final_status: str  # "completed" | "incomplete" | "never_started"
    wait_time_total: float
    preempted_count: int


@dataclass(frozen=True)
class SchedulerResult:
    jobs: tuple[JobOutcome, ...]
    pool_total_gpus: int
    makespan: float
    gpu_utilization_pct: float


@dataclass
class _RunningJob:
    job: SchedJob
    gpus_held: int
    remaining_duration: float
    resumed_at: float
    generation: int  # invalidates this job's previously-scheduled COMPLETION event if it's preempted and re-placed


@dataclass
class _JobState:
    job: SchedJob
    remaining_duration: float
    segments: list[TimelineSegment] = field(default_factory=list)
    wait_time_total: float = 0.0
    preempted_count: int = 0
    waiting_since: float = 0.0
    final_status: str = "never_started"


def simulate_scheduler(
    jobs: list[SchedJob],
    pool: GpuPool,
    preemption_enabled: bool = True,
    horizon: float | None = None,
) -> SchedulerResult:
    if not jobs:
        raise ValueError("At least one job is required")
    ids = [j.job_id for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("job_id values must be unique")
    oversized = [j.job_id for j in jobs if j.gpu_count > pool.total_count]
    if oversized:
        raise ValueError(f"Job(s) request more GPUs than the pool has: {oversized}")

    order_index = {j.job_id: i for i, j in enumerate(jobs)}  # final, fully-deterministic tiebreaker
    states: dict[str, _JobState] = {
        j.job_id: _JobState(job=j, remaining_duration=j.duration, waiting_since=j.submit_time) for j in jobs
    }
    running: dict[str, _RunningJob] = {}
    queued: set[str] = set()
    free_gpus = pool.total_count
    generation_counter = 0

    heap: list[tuple[float, int, int, float, int, str, int]] = []

    def push_event(time: float, etype: int, job_id: str, generation: int = 0) -> None:
        j = states[job_id].job
        # Sort key: time asc, event type asc (ARRIVAL before COMPLETION at
        # equal time — let a finishing job free capacity before deciding
        # whether a same-instant arrival must queue), priority desc,
        # submit_time asc, list order asc — the last two make same-priority
        # ties fully deterministic.
        heapq.heappush(heap, (time, etype, -j.priority, j.submit_time, order_index[job_id], job_id, generation))

    for j in jobs:
        push_event(j.submit_time, _ARRIVAL, j.job_id)

    def queue_sort_key(job_id: str) -> tuple[int, float, int]:
        j = states[job_id].job
        return (-j.priority, j.submit_time, order_index[job_id])

    def victim_sort_key(job_id: str) -> tuple[int, int, float]:
        rj = running[job_id]
        # Prefer preempting: lowest priority first, then smallest job (frees
        # just enough without over-preempting), then most-recently-started
        # (least sunk progress lost).
        return (rj.job.priority, rj.gpus_held, -rj.resumed_at)

    def place(job_id: str, time: float) -> None:
        nonlocal free_gpus, generation_counter
        st = states[job_id]
        free_gpus -= st.job.gpu_count
        generation_counter += 1
        running[job_id] = _RunningJob(
            job=st.job, gpus_held=st.job.gpu_count, remaining_duration=st.remaining_duration,
            resumed_at=time, generation=generation_counter,
        )
        st.wait_time_total += time - st.waiting_since
        queued.discard(job_id)
        push_event(time + st.remaining_duration, _COMPLETION, job_id, generation_counter)

    def preempt(job_id: str, time: float) -> None:
        nonlocal free_gpus
        rj = running.pop(job_id)
        st = states[job_id]
        elapsed = time - rj.resumed_at
        st.segments.append(TimelineSegment(start=rj.resumed_at, end=time))
        st.remaining_duration = max(0.0, rj.remaining_duration - elapsed)
        st.preempted_count += 1
        st.waiting_since = time
        free_gpus += rj.gpus_held
        queued.add(job_id)

    def try_admit(job_id: str, time: float) -> bool:
        st = states[job_id]
        need = st.job.gpu_count
        if need <= free_gpus:
            place(job_id, time)
            return True
        if not preemption_enabled:
            return False

        candidates = sorted((jid for jid in running if running[jid].job.priority < st.job.priority), key=victim_sort_key)
        freed = free_gpus
        to_preempt: list[str] = []
        for jid in candidates:
            if freed >= need:
                break
            to_preempt.append(jid)
            freed += running[jid].gpus_held
        if freed < need:
            return False

        for jid in to_preempt:
            preempt(jid, time)
        place(job_id, time)
        return True

    def sweep_queue(time: float) -> None:
        # Greedy pass in priority order; a job that doesn't fit is skipped,
        # not blocking — a smaller lower-priority job behind it may still fit.
        for job_id in sorted(queued, key=queue_sort_key):
            if job_id in queued and states[job_id].job.gpu_count <= free_gpus:
                place(job_id, time)

    horizon_hit = False
    while heap:
        time, etype, _, _, _, job_id, gen = heapq.heappop(heap)
        if horizon is not None and time > horizon:
            horizon_hit = True
            break

        if etype == _COMPLETION:
            rj = running.get(job_id)
            if rj is None or rj.generation != gen:
                continue  # stale event for a job that's since been preempted and re-placed
            running.pop(job_id)
            free_gpus += rj.gpus_held
            st = states[job_id]
            st.segments.append(TimelineSegment(start=rj.resumed_at, end=time))
            st.final_status = "completed"
            sweep_queue(time)
        else:
            if not try_admit(job_id, time):
                queued.add(job_id)

    if running:
        # Only reachable when `horizon` cut the simulation short — close out
        # each still-running job's open segment at the cutoff.
        cutoff = horizon if horizon_hit else max(rj.resumed_at for rj in running.values())
        for job_id, rj in list(running.items()):
            states[job_id].segments.append(TimelineSegment(start=rj.resumed_at, end=cutoff))

    for st in states.values():
        if st.final_status != "completed":
            st.final_status = "incomplete" if st.segments else "never_started"

    makespan = max((seg.end for st in states.values() for seg in st.segments), default=0.0)
    total_gpu_time = pool.total_count * makespan
    busy_gpu_time = sum((seg.end - seg.start) * st.job.gpu_count for st in states.values() for seg in st.segments)
    gpu_utilization_pct = (busy_gpu_time / total_gpu_time * 100) if total_gpu_time > 0 else 0.0

    outcomes = tuple(
        JobOutcome(
            job_id=st.job.job_id,
            team=st.job.team,
            segments=tuple(st.segments),
            final_status=st.final_status,
            wait_time_total=st.wait_time_total,
            preempted_count=st.preempted_count,
        )
        for st in states.values()
    )

    return SchedulerResult(
        jobs=outcomes,
        pool_total_gpus=pool.total_count,
        makespan=makespan,
        gpu_utilization_pct=gpu_utilization_pct,
    )

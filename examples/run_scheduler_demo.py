from hydra_aixia.scheduler_dvfs import Task, hybrid_edf_dvfs_schedule

if __name__ == "__main__":
    tasks = [
        Task(task_id="t1", workload=1.0, deadline=3.0),
        Task(task_id="t2", workload=2.0, deadline=5.0),
        Task(task_id="t3", workload=1.5, deadline=4.0),
    ]
    for decision in hybrid_edf_dvfs_schedule(tasks):
        print(decision)

from effective_hsmt.regression import run_all_benchmarks, package_status

rows = run_all_benchmarks()
for name, ok, detail in rows:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
status = package_status()
print(f"\nPassed {status['passed']}/{status['total']} checks.")
print("Foundation:", status["foundation"])
print("Channel map included:", status["channel_map_included"])
print("Measure selected by calculus:", status["measure_selected_by_calculus"])

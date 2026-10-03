# Prospective AS02 rerun

The user requested a new initial policy followed by experiments. POLICY_V1.md and the 37-file PRE_RUN_LOCK.json were frozen before any new inference. Historical experiments are explicitly acknowledged; no dates or prior history were rewritten.

The first attempt failed before inference because the sandbox could not access Metal. The unchanged GPU-enabled retry writes to runs/policy_first_gpu/. Check ATTEMPT_2_STATUS.json and execution_attempt_2.log for live status. Do not restart or overwrite an existing attempt. Local runtime: /private/tmp/as02-local-runtime/bin/python. The model weights remain in the original workspace's .models/phi-4-mini-instruct-4bit folder, not this deliverable.

After the retry completes successfully:

1. Run finalize_evidence.py using the local runtime. It refuses incomplete runs, checks chronology and all frozen hashes, recomputes main metrics, audits paired safety metrics, verifies exact human-label transfers, and only then writes POLICY_V2.md.
2. Run the PDF artifact marker for two created PDFs, then build_updated_report.py using the bundled document/PDF runtime (ReportLab and pypdf). It creates the prospective addendum and a combined updated AS02 report, preserving the historical report as a labelled part.
3. Render and inspect the resulting PDFs, fixing only document layout if needed. Never change V1, data, model-run source or saved responses after the freeze.
4. Run package_updated.py and verify both ZIPs. It produces versioned team and AS02 handoffs while retaining original source packages.
5. Copy the verified files to the existing user-facing outputs directory with required filesystem permission. Do not overwrite the September 28 deliverables. No Canvas submission is authorized by this workflow.

The original 17 boundary tests were rerun successfully and are recorded in boundary_tests.txt. Additional semantic human review and new human cost timing are not invented. Any changed safety output retains null human labels unless actually reviewed.

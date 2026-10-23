# Step 1 prompt for an AI coding assistant (on your laptop)

Paste this into an AI coding assistant. It builds the folder that goes to the server, and tests it on the CPU first.

```
Build a deployment folder for The EYE v3.5 and test it on the CPU. Do not change or move any existing project file.
1. Create PROJECT - 1\EYE_DEPLOY with this layout:
   EYE_DEPLOY\m2_eye_app_v35.py      (copy of Milestone 2\M2 extra\m2_eye_app_v35.py)
   EYE_DEPLOY\eye.service            (copy of Downloads\eye.service)
   EYE_DEPLOY\project\               (copy of the folder start_eye_v35.bat passes as M2_PROJECT_DIR, i.e. Milestone 3\M3 HOST)
2. Inside EYE_DEPLOY\project, make sure everything the EYE reads is present, keeping each file's path from its anchor folder:
   - every detector's weights listed in milestone2_outputs\detect\results.json and in each "M2 extra\boost*\train.json"
     (copy to project\<the path from "M2 extra" or "milestone2_outputs" onwards>);
   - the classifier: project\milestone1_outputs_v2\experiments\E_convnext_tiny_finetune\best.pt;
   - M2 extra\selected_model.json and the file its "selection" field names; the boost*\tests scenes; object_list.csv;
     milestone2_outputs\boxes_master.csv and per-class files; the whole "Dataset - Updated" photo folder.
   Leave out training runs, caches, checkpoints other than best.pt, and anything the EYE does not read.
3. Path check (Linux is case-sensitive): for every weights path in those JSON files, take the part from its anchor folder
   onwards and confirm that file exists inside EYE_DEPLOY\project with exactly that spelling and case. Also confirm every
   Dataset - Updated\images_<OBJECT_ID> folder name matches the object IDs in boxes_master.csv exactly. Report any mismatch.
4. Write EYE_DEPLOY\requirements.txt from the EYE's actual imports, with the versions installed here but WITHOUT local
   suffixes such as +cu121 (the server is a CPU-only ARM machine). Use opencv-python-headless instead of opencv-python
   if OpenCV is needed. Report the file.
5. CPU test: from EYE_DEPLOY, with CUDA_VISIBLE_DEVICES set to -1, M2_PUBLIC=1, M2_NO_BROWSER=1, M2_PORT=7869 and
   M2_PROJECT_DIR=EYE_DEPLOY\project, start the EYE. Report /api/config: models (all seven detectors must be listed:
   Milestone 2 Runs 1-5, yolo26s_boost and yolo26s_boost2), panel (the COMPOUND members), panel_cls, n_test_photos, layouts,
   stress_max. If any of the seven is missing, find why before going on. Then for a 3x3 grid (seed 7): time one detection with yolo26s_boost, one COMPOUND, and one classifier
   detection on a single photo (seed 7). Stop only that server.
6. Zip EYE_DEPLOY into PROJECT - 1\EYE_DEPLOY.zip (paths inside the zip relative to EYE_DEPLOY). Report its size and
   the size of each top-level folder. STOP.
```

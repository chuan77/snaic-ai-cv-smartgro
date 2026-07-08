"""Pushes actively-captured, uncertain checkout frames into a Label Studio project for review."""
from src.deploy.active_learning_capture import get_capture_dir
from src.deploy.label_studio_push import get_ls_project_id, push_staging_dir

if __name__ == "__main__":
    project_id = get_ls_project_id()
    if not project_id:
        raise SystemExit("SMARTCART_LS_PROJECT_ID must be set to push captures to a Label Studio project.")

    pushed = push_staging_dir(get_capture_dir(), project_id)
    print(f"Pushed {pushed} captured frames to Label Studio project {project_id}.")

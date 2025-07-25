-- 1) fetch_pending_task 시그니처 조회
SELECT
  p.proname    AS function_name,
  pg_catalog.pg_get_function_arguments(p.oid) AS arguments
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'public'
  AND p.proname = 'fetch_pending_task';


  -- 2) fetch_done_data 시그니처 조회
SELECT
  p.proname    AS function_name,
  pg_catalog.pg_get_function_arguments(p.oid) AS arguments
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'public'
  AND p.proname = 'fetch_done_data';


  -- 3) save_task_result 시그니처 조회
SELECT
  p.proname    AS function_name,
  pg_catalog.pg_get_function_arguments(p.oid) AS arguments
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_namespace n ON p.pronamespace = n.oid
WHERE n.nspname = 'public'
  AND p.proname = 'save_task_result';


DROP FUNCTION IF EXISTS public.fetch_done_data(text, text);


kubectl logs -f agent-monitoring-deployment-56c6955f97-rcljv > output.log 2>&1
kubectl get pods -l app=agent-monitoring
uv run main.py > output.log 2>&1
kubectl logs -f agent-monitoring-deployment-5bbbb76878-vgm7m > output.log 2>&1
kubectl logs -f agent-monitoring-deployment-69699c5bc9-k9s7s > output.log 2>&1
python main.py > output.log 2>&1

uv venv
uv pip install -r requirements.txt
source .venv/Scripts/activate
deactivate
"""Regression tests for the concise Swagger UI contract."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_openapi_uses_seven_non_duplicated_groups():
    schema = client.get("/openapi.json").json()

    assert [tag["name"] for tag in schema["tags"]] == [
        "区域",
        "Patch",
        "Embedding",
        "任务结果",
        "自定义模型",
        "系统模型",
        "SAM3",
    ]
    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            assert len(operation.get("tags", [])) == 1


def test_operation_descriptions_do_not_repeat_native_swagger_tables():
    schema = client.get("/openapi.json").json()

    for path, path_item in schema["paths"].items():
        for method, operation in path_item.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"}:
                continue
            description = operation.get("description", "")
            assert "### 参数填写说明" not in description
            assert "### Request Body 说明" not in description
            assert "### Response 说明" not in description
            assert "| 参数 |" not in description
            assert len(description) <= 600


def test_concise_docs_keep_chinese_field_help_and_runnable_examples():
    schema = client.get("/openapi.json").json()
    create_model = schema["paths"]["/models"]["post"]
    region_parameter = schema["paths"]["/regions/{region_id}"]["get"]["parameters"][0]

    request_schema = create_model["requestBody"]["content"]["application/json"]["schema"]
    request_name = request_schema["$ref"].rsplit("/", 1)[-1]
    properties = schema["components"]["schemas"][request_name]["properties"]

    assert region_parameter["example"] == "haidian"
    assert "训练方式" in properties["training_method"]["description"]
    assert "默认" in properties["epochs"]["description"]


def test_task_result_exposes_one_clear_time_contract():
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][
        "/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result"
    ]["get"]
    parameters = {item["name"]: item for item in operation["parameters"]}

    assert "period" not in parameters
    assert not parameters["version"].get("required", False)
    assert set(parameters) >= {"month", "before_month", "after_month"}
    assert "YYYYMM" in parameters["month"]["description"]
    assert "YYYY-MM" in parameters["month"]["description"]
    assert "YYYYMM" in parameters["before_month"]["description"]
    assert "YYYY-MM" in parameters["after_month"]["description"]

    version_examples = parameters["version"]["examples"]
    assert version_examples["haidian_p10c"]["value"] == "v1"
    assert version_examples["harbin_v5"]["value"] == "v2"


def test_task_result_documents_region_task_options_from_task_list():
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][
        "/regions/{region_id}/patches/{patch_id}/tasks/{task_type}/result"
    ]["get"]
    task_parameter = next(
        item for item in operation["parameters"] if item["name"] == "task_type"
    )
    documented = task_parameter["x-region-task-options"]

    for region_id in ("haidian", "harbin"):
        response = client.get(f"/regions/{region_id}/tasks")
        listed = [task["id"] for task in response.json()["tasks"]]
        assert documented[region_id] == listed

    assert "GET /regions/{region_id}/tasks" in task_parameter["description"]
    assert "change_detection" in documented["haidian"]
    assert "change_detection" in documented["harbin"]
    assert "construction" in documented["haidian"]
    assert "construction" not in documented["harbin"]


def test_task_summary_uses_the_same_clear_time_contract():
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][
        "/regions/{region_id}/tasks/{task_type}/summary"
    ]["get"]
    parameters = {item["name"]: item for item in operation["parameters"]}

    assert "period" not in parameters
    assert not parameters["version"].get("required", False)
    assert set(parameters) >= {"month", "patch_ids"}
    assert "before_month" not in parameters
    assert "after_month" not in parameters
    assert "patch_000000" in parameters["patch_ids"]["description"]
    assert parameters["patch_ids"]["examples"]["single"]["value"] == ["patch_000000"]
    assert parameters["patch_ids"]["examples"]["multiple"]["value"] == [
        "patch_000000",
        "patch_000001",
    ]


def test_change_summary_has_its_own_two_month_contract():
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][
        "/regions/{region_id}/change-detection/summary"
    ]["get"]
    parameters = {item["name"]: item for item in operation["parameters"]}

    assert set(parameters) >= {"before_month", "after_month", "patch_ids"}
    assert "month" not in parameters
    assert "task_type" not in parameters


def test_custom_model_analysis_is_not_exposed():
    schema = client.get("/openapi.json").json()
    assert "/models/{model_id}/analysis" not in schema["paths"]


def test_embedding_version_is_optional_and_documents_region_defaults():
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][
        "/regions/{region_id}/patches/{patch_id}/embedding"
    ]["get"]
    version = next(item for item in operation["parameters"] if item["name"] == "version")

    assert not version.get("required", False)
    assert version["examples"]["haidian_p10c"]["value"] == "v1"
    assert version["examples"]["harbin_v5"]["value"] == "v2"

    month = next(item for item in operation["parameters"] if item["name"] == "month")
    assert not month.get("required", False)
    assert "哈尔滨" in month["description"]
    assert "2025-04" in month["description"]
    assert "2026-05" in month["description"]
    assert "海淀" in month["description"]
    assert month["examples"]["harbin_v5"]["value"] == "202510"
    assert month["examples"]["haidian_p10c"]["value"] == "202512"
    assert "example" not in month


def test_online_mosaic_route_is_not_exposed():
    schema = client.get("/openapi.json").json()
    assert "/regions/{region_id}/mosaic" not in schema["paths"]


def test_binary_download_routes_do_not_claim_json_success_bodies():
    schema = client.get("/openapi.json").json()
    paths = (
        "/models/results/{filename}",
        "/system-models/results/{filename}",
    )
    for path in paths:
        media = schema["paths"][path]["get"]["responses"]["200"]["content"]
        assert "application/json" not in media


def test_unimplemented_xyz_tile_route_is_not_advertised():
    schema = client.get("/openapi.json").json()
    assert (
        "/regions/{region_id}/tasks/{task_type}/tiles/{z}/{x}/{y}.png"
        not in schema["paths"]
    )

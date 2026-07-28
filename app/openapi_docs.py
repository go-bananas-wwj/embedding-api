"""OpenAPI documentation helpers for the concise Swagger UI.

FastAPI and Swagger already render parameters, request bodies, responses and
schemas. This module enriches those native panels with Chinese field help and
runnable examples without repeating the same information as Markdown tables.
"""

from typing import Any, Dict, Iterable, List, Optional


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def enhance_openapi_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate and return a generated OpenAPI schema with readable docs."""
    components = schema.get("components", {}).get("schemas", {})
    _enhance_component_descriptions(components)
    for path, path_item in schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            _enhance_operation(operation, method.upper(), path, components)
    return schema


def _enhance_operation(
    operation: Dict[str, Any],
    method: str,
    path: str,
    components: Dict[str, Any],
) -> None:
    if operation.get("x-docs-enhanced"):
        return

    parameters = operation.get("parameters", [])
    _apply_preferred_parameter_examples(parameters, path)
    _apply_region_task_options(parameters, path)
    _enhance_response_descriptions(operation.get("responses", {}), components)
    operation["x-docs-enhanced"] = True
    operation.setdefault("operationId", _operation_id(method, path))


def _apply_region_task_options(
    parameters: List[Dict[str, Any]],
    path: str,
) -> None:
    """Document task choices from the same config used by the task-list API."""
    if "{region_id}" not in path:
        return
    task_parameter = next(
        (param for param in parameters if param.get("name") == "task_type"),
        None,
    )
    if task_parameter is None:
        return

    # Import lazily to keep the OpenAPI helper independent during module import.
    from app.config import get_config

    config = get_config()
    region_options: Dict[str, List[str]] = {}
    examples: Dict[str, Dict[str, str]] = {}
    for region_id in ("haidian", "harbin"):
        region = config.get_region(region_id) or {}
        region_name = region.get("name", region_id)
        tasks = region.get("tasks") or {}
        region_options[region_id] = list(tasks)
        for task_id, task in tasks.items():
            examples[f"{region_id}_{task_id}"] = {
                "summary": f"{region_name}：{task.get('name', task_id)}",
                "value": task_id,
            }

    task_parameter["x-region-task-options"] = region_options
    task_parameter["examples"] = examples
    base = (task_parameter.get("description") or "").strip()
    guidance = (
        "可选值按区域不同；请先调用 `GET /regions/{region_id}/tasks`，"
        "并直接使用返回的 `tasks[].id`。下方示例名称已标明适用区域。"
    )
    if guidance not in base:
        task_parameter["description"] = f"{base} {guidance}".strip()
    schema = task_parameter.get("schema")
    if isinstance(schema, dict):
        schema["description"] = task_parameter["description"]


def _parameter_section(parameters: List[Dict[str, Any]]) -> str:
    if not parameters:
        return ""

    rows = [
        "| 参数 | 位置 | 必填 | 默认值/范围 | 怎么填 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for param in parameters:
        schema = param.get("schema", {})
        rows.append(
            "| `{name}` | {location} | {required} | {limits} | {example}<br>{description} |".format(
                name=param.get("name", ""),
                location=_location_label(param.get("in", "")),
                required="是" if param.get("required") else "否",
                limits=_schema_limits(schema),
                example=_parameter_example_text(param),
                description=_field_description(
                    param.get("name", ""),
                    param.get("description"),
                ),
            )
        )
    return "### 参数填写说明\n\n" + "\n".join(rows)


def _apply_preferred_parameter_examples(
    parameters: List[Dict[str, Any]],
    path: str,
) -> None:
    """Make Swagger UI Try-it-out boxes prefer runnable examples.

    FastAPI can emit multiple examples, but Swagger UI often picks the first or
    leaves optional fields blank. For frontend handoff, prefer examples that
    work against the checked-in Haidian latest data.
    """
    preferred = {
        "region_id": "haidian",
        "patch_id": "patch_000000",
        "task_type": "building_extraction",
        "month": "202512",
        "date": "202512",
        "sensor_type": "s2",
        "format": "png",
        "filename": "patch_000000.png",
    }
    if "/sam3/" in path:
        preferred["region_id"] = "haidian"
    if path.endswith("/embedding"):
        # The two regions have different month ranges. Leaving this optional
        # lets the endpoint select each region's latest available month.
        preferred.pop("month", None)
    if "/tasks/" in path:
        preferred["region_id"] = "haidian"
        preferred["task_type"] = "building_extraction"
    for param in parameters:
        name = param.get("name")
        if name not in preferred:
            continue
        value = preferred[name]
        schema = param.get("schema")
        if isinstance(schema, dict):
            examples = schema.get("examples")
            if value is not None:
                schema["example"] = value
                if not examples:
                    schema["examples"] = [value]
            elif "example" in schema:
                schema.pop("example", None)
        if value is not None:
            param["example"] = value


def _request_body_section(
    request_body: Optional[Dict[str, Any]],
    components: Dict[str, Any],
) -> str:
    if not request_body:
        return ""

    content = request_body.get("content", {})
    if not content:
        return "### Request Body 说明\n\n请求体未声明具体 content type。"

    parts = ["### Request Body 说明"]
    for content_type, media in content.items():
        schema = _resolve_schema(media.get("schema", {}), components)
        fields = _schema_fields(schema, components)
        parts.append(f"\n**Content-Type**: `{content_type}`")
        if not fields:
            parts.append("\n请求体 schema 未展开字段；请参考下方 Swagger Schema 面板。")
            continue
        rows = [
            "",
            "| 字段 | 必填 | 类型/默认值/范围 | 怎么填 |",
            "| --- | --- | --- | --- |",
        ]
        for field in fields:
            rows.append(
                "| `{name}` | {required} | {limits} | {description} |".format(
                    name=field["name"],
                    required="是" if field["required"] else "否",
                    limits=_schema_limits(field["schema"]),
                    description=_field_description(
                        field["name"],
                        field["schema"].get("description"),
                    ),
                )
            )
        parts.extend(rows)
    return "\n".join(parts)


def _response_section(
    responses: Dict[str, Any],
    components: Dict[str, Any],
) -> str:
    if not responses:
        return ""

    rows = [
        "### Response 说明",
        "",
        "| 状态码 | Content-Type | 返回内容 |",
        "| --- | --- | --- |",
    ]
    for status_code, response in sorted(responses.items(), key=lambda item: item[0]):
        content = response.get("content", {})
        if not content:
            rows.append(
                f"| `{status_code}` | 无 | {_status_description(status_code, response)} |"
            )
            continue
        media_types = []
        payloads = []
        for content_type, media in content.items():
            media_types.append(f"`{content_type}`")
            payloads.append(_schema_title(media.get("schema", {}), components))
        rows.append(
            "| `{}` | {} | {} |".format(
                status_code,
                "<br>".join(media_types),
                "<br>".join(payloads) or _status_description(status_code, response),
            )
        )
    return "\n".join(rows)


def _set_request_body_description(request_body: Dict[str, Any], section: str) -> None:
    existing = (request_body.get("description") or "").strip()
    if "Request Body 说明" in existing:
        return
    request_body["description"] = f"{existing}\n\n{section}".strip()


def _enhance_response_descriptions(
    responses: Dict[str, Any],
    components: Dict[str, Any],
) -> None:
    for status_code, response in responses.items():
        existing = response.get("description", "")
        if existing and existing != "Successful Response":
            continue
        if str(status_code).startswith("2"):
            response["description"] = "成功响应。返回内容见下方 Schema / Example。"
        elif status_code == "422":
            response["description"] = "参数校验失败。请检查必填字段、类型、取值范围和请求体格式。"
        elif status_code == "404":
            response["description"] = "资源不存在。请检查 region、patch、task、model 或文件名是否正确。"
        elif status_code == "501":
            response["description"] = "接口已声明但当前版本尚未实现。"
        else:
            response["description"] = _status_description(str(status_code), response)


def _enhance_component_descriptions(components: Dict[str, Any]) -> None:
    """Localize common schema property descriptions for Swagger's Schema panel."""
    for schema in components.values():
        properties = schema.get("properties", {})
        for name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            prop["description"] = _field_description(name, prop.get("description"))


def _schema_fields(
    schema: Dict[str, Any],
    components: Dict[str, Any],
) -> List[Dict[str, Any]]:
    schema = _resolve_schema(schema, components)
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    result = []
    for name, field_schema in properties.items():
        resolved = _resolve_schema(field_schema, components)
        result.append(
            {
                "name": name,
                "required": name in required,
                "schema": resolved,
            }
        )
    return result


def _resolve_schema(
    schema: Dict[str, Any],
    components: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    ref = schema.get("$ref")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        return components.get(name, schema)
    for key in ("allOf", "anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            merged = dict(schema)
            first = _resolve_schema(variants[0], components)
            merged.update({k: v for k, v in first.items() if k not in merged})
            return merged
    return schema


def _schema_limits(schema: Dict[str, Any]) -> str:
    schema = schema or {}
    parts = []
    type_text = _schema_type(schema)
    if type_text:
        parts.append(type_text)
    if "default" in schema:
        parts.append(f"默认 `{_format_value(schema['default'])}`")
    enum_values = schema.get("enum")
    if enum_values:
        parts.append("可选 " + " / ".join(f"`{v}`" for v in enum_values))
    ranges = []
    if "minimum" in schema:
        ranges.append(f">= {schema['minimum']}")
    if "maximum" in schema:
        ranges.append(f"<= {schema['maximum']}")
    if ranges:
        parts.append("范围 " + "，".join(ranges))
    lengths = []
    if "minLength" in schema:
        lengths.append(f"长度 >= {schema['minLength']}")
    if "maxLength" in schema:
        lengths.append(f"长度 <= {schema['maxLength']}")
    if "minItems" in schema:
        lengths.append(f"数量 >= {schema['minItems']}")
    if "maxItems" in schema:
        lengths.append(f"数量 <= {schema['maxItems']}")
    if lengths:
        parts.append("；".join(lengths))
    return "<br>".join(parts) if parts else "见说明"


def _parameter_example_text(param: Dict[str, Any]) -> str:
    example = param.get("example")
    schema = param.get("schema") or {}
    if example is None:
        example = schema.get("example")
    if example is None:
        examples = schema.get("examples")
        if isinstance(examples, list) and examples:
            example = examples[0]
    if example is None:
        return "示例：按需留空或参考说明。"
    return f"示例：`{_format_value(example)}`。"


def _schema_type(schema: Dict[str, Any]) -> str:
    if "anyOf" in schema:
        variants = [_schema_type(s) for s in schema.get("anyOf", [])]
        variants = [v for v in variants if v and v != "null"]
        return " / ".join(dict.fromkeys(variants))
    schema_type = schema.get("type")
    if schema_type == "array":
        items = schema.get("items", {})
        item_type = _schema_type(items) or "any"
        return f"array<{item_type}>"
    if schema_type:
        return str(schema_type)
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    return ""


def _schema_title(schema: Dict[str, Any], components: Dict[str, Any]) -> str:
    schema = _resolve_schema(schema, components)
    if "$ref" in schema:
        return _response_schema_label(schema["$ref"].rsplit("/", 1)[-1])
    if schema.get("title"):
        return _response_schema_label(schema["title"])
    if schema.get("type") == "array":
        return "数组响应"
    if schema.get("type") == "string":
        return "二进制或字符串响应"
    return "响应体见 Schema"


def _clean_description(value: Optional[str]) -> str:
    if not value:
        return "按接口说明填写。"
    text = str(value).replace("\n", " ").replace("|", "\\|").strip()
    return text


def _field_description(name: str, value: Optional[str]) -> str:
    """Return a Chinese-first field description for generated tables."""
    key = name or ""
    if key in FIELD_DOCS:
        return FIELD_DOCS[key]
    text = _clean_description(value)
    if _contains_cjk(text):
        return text
    if text == "按接口说明填写。":
        return text
    return f"请按接口说明填写。原始说明：{text}"


def _status_description(status_code: str, response: Dict[str, Any]) -> str:
    detail = response.get("description")
    if detail and detail != "Successful Response":
        return _clean_description(detail)
    if str(status_code).startswith("2"):
        return "请求成功。"
    if status_code == "422":
        return "参数校验失败。"
    if status_code == "404":
        return "资源不存在。"
    if status_code == "501":
        return "当前版本尚未实现。"
    return "错误响应。"


def _location_label(location: str) -> str:
    return {
        "path": "路径参数",
        "query": "查询参数",
        "header": "Header",
        "cookie": "Cookie",
    }.get(location, location or "-")


def _operation_id(method: str, path: str) -> str:
    safe = path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
    safe = safe.replace(".", "_").replace("-", "_") or "root"
    return f"{method.lower()}_{safe}"


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _response_schema_label(name: str) -> str:
    labels = {
        "HealthResponse": "健康检查响应（HealthResponse）",
        "RegionsResponse": "区域列表响应（RegionsResponse）",
        "RegionDetail": "区域详情响应（RegionDetail）",
        "PaginatedPatchesResponse": "分页 Patch 列表响应（PaginatedPatchesResponse）",
        "PatchDetail": "Patch 详情响应（PatchDetail）",
        "EmbeddingStats": "Embedding 统计响应（EmbeddingStats）",
        "TasksResponse": "任务列表响应（TasksResponse）",
        "TaskSummary": "任务统计摘要响应（TaskSummary）",
        "TilesResponse": "瓦片列表响应（TilesResponse）",
        "ModelOut": "模型详情响应（ModelOut）",
        "InferResult": "推理结果 URL 响应（InferResult）",
        "BatchInferResponse": "批量推理响应（BatchInferResponse）",
        "JobStatusOut": "训练任务状态响应（JobStatusOut）",
        "SegmentResponse": "SAM3 GeoJSON 分割响应（SegmentResponse）",
        "EmbedResponse": "SAM3 embedding 响应（EmbedResponse）",
        "StatusResponse": "SAM3 状态响应（StatusResponse）",
        "HTTPValidationError": "参数校验错误响应（HTTPValidationError）",
        "ErrorResponse": "错误响应（ErrorResponse）",
    }
    return labels.get(name, name)


FIELD_DOCS = {
    "region_id": "区域 ID。可选值通常为 `harbin` 或 `haidian`；用于指定数据所属区域。",
    "patch_id": "Patch 编号。格式为 `patch_000000`；用于定位单个遥感切片。",
    "page": "页码。从 1 开始；用于分页获取列表。",
    "page_size": "每页数量。默认通常为 20，最大通常为 100；用于控制单次返回条数。",
    "bbox": "WGS84 边界框过滤条件，格式为 `min_lon,min_lat,max_lon,max_lat`。",
    "footprint_wgs84": "Patch 的真实 WGS84 GeoJSON 四边形边界。前端绘制边框时优先使用它，不要用外接矩形代替。",
    "format": "返回格式。按接口可选 `png`、`npy`、`json`、`cache` 或 `tif`。",
    "version": "数据或模型版本。按接口可选 `v1`、`v2`；系统模型不传时会按区域自动选择可用版本，海淀最新模型通常使用 `v1`。",
    "month": "影像月份或日期。支持 `YYYY-MM`、`YYYYMM` 或部分接口的 `YYYYMMDD`；日级表示精确日期，月级请求会按日期倒序选择同月最新的一景。",
    "date": "影像日期或月份。支持 `YYYY-MM`、`YYYYMM` 或 `YYYYMMDD`；`YYYYMMDD` 精确命中，月级请求会按日期倒序选择同月最新的一景。",
    "sensor_type": "传感器类型。可选 `s2`、`s1`、`landsat`、`highres`；`highres` 用于带地理参考的高分辨率 RGB 光学 GeoTIFF。",
    "patch_ids": "Patch 编号数组。每项格式为 `patch_000000`，批量推理最多 100 个。使用自定义多类别 model_id 时，后端会对每个 Patch 自动运行模型绑定的全部类别头，不需要再次传 class_ids、training_method 或 head_type。",
    "task_type": "任务类型，例如 `building_extraction`、`water_extraction`、`change_detection`。自定义训练的 GeoJSON Feature 中可不传；不传时后端按模型类型推导默认任务。",
    "task_id": "系统模型任务 ID，例如 `building_extraction`、`water_extraction`。",
    "model_id": "模型 ID。自定义模型形如 `model_xxxxxxxx`；系统模型可直接使用任务 ID。",
    "job_id": "训练任务 ID。创建模型后返回，用于轮询训练状态。",
    "filename": "文件名。必须使用接口返回的文件名，不要手写路径或包含目录分隔符。",
    "period": "任务时间段。变化检测通常为 `before_vs_after`，例如 `2025-04_vs_2025-06`。",
    "before_month": "变化检测起始月份，例如 `2025-04`。",
    "after_month": "变化检测结束月份，例如 `2025-06`。",
    "z": "XYZ 瓦片缩放级别。当前 XYZ 瓦片接口为预留接口。",
    "x": "XYZ 瓦片 X 坐标。当前 XYZ 瓦片接口为预留接口。",
    "y": "XYZ 瓦片 Y 坐标。当前 XYZ 瓦片接口为预留接口。",
    "name": "名称。用于模型、类别或展示项的人类可读名称。",
    "model_type": "训练类型。可选 `single_time_detection`（单时间检测）或 `change_detection`（双时相变化检测）；旧值 `classification` 仍兼容。",
    "training_method": "训练方式。默认 `xuannv_earth`；`traditional_ml` 使用 Sentinel-2 六波段+四个光谱指数，每个有标注类别训练一个 Random Forest；`aef` 使用年度 embedding，当前任意月份固定回退到 2025 年；`dinov3_sat493m` 使用对应月份光学影像，每个有标注类别训练一个像素 MLP。多类别最终仍返回一个 model_id。实际资产可用性以 `GET /models/capabilities` 为准。",
    "requested_training_method": "前端请求的训练方式；旧模型和系统模型可能为空。",
    "resolved_training_method": "后端实际执行的算法，例如 `pu_query_retrieval`、`binary_conv3x3` 或 `random_forest`。",
    "feature_source": "训练特征来源，例如 `xuannv_embedding` 或 `sentinel2_l2a`。",
    "embedding_version": "训练/推理使用的 embedding 版本。按区域可选 `v1` 或 `v2`。",
    "epochs": "训练迭代次数。默认 100，范围 1~1000。后端按类别分别统计有效 Polygon：某类达到 10 个时训练 Binary Conv 3x3，服务端最多执行 100 轮；某类少于 10 个时使用免迭代的 PU + Query，此参数对该类不参与计算。",
    "class_ids": "候选类别 ID 列表。后端以 annotations 中实际出现的 class_id 为准：有标注的类别各自训练一个二分类头，没有 Polygon 标注的类别自动跳过；前端请求格式不随训练策略变化。",
    "description": "描述文本。用于补充说明模型、任务或配置用途。",
    "annotations": "GeoJSON 标注包。坐标必须是 WGS84，经纬度顺序为 `[lon, lat]`；Polygon 内部是对应类别的正样本，外部是未标注样本而不是直接负样本。后端按 class_id 独立计数和训练：某类少于 10 个 Polygon 使用 PU + Query，大于等于 10 个使用 Binary Conv 3x3；MultiPolygon 的每个独立 Polygon 分别计数。",
    "classes": "类别定义列表。每个类别包含 `id`、`name`、`color`。",
    "id": "资源 ID。用于唯一标识区域、模型、任务或类别。",
    "color": "颜色值。建议使用十六进制格式，例如 `#FF0000`。",
    "class_id": "类别 ID。必须能在 `classes` 中找到对应定义。",
    "class_name": "类别显示名称。用于前端展示。",
    "geometry": "GeoJSON 几何对象。当前训练标注支持 Polygon 或 MultiPolygon。",
    "point_coords": "WGS84 点击点列表。每个点格式为 `[经度, 纬度]`。",
    "point_labels": "可选点标签。`1` 表示前景目标点，`0` 表示背景排除点；不传时默认全为 `1`。",
    "multimask_output": "是否返回多个候选结果。默认 `false`；常规前端交互建议保持默认。",
    "include_masks": "是否额外返回 base64 PNG mask。默认 `false`；只需要标注框时不要开启。",
    "source_scene": "实际参与计算的原始影像文件名 stem，例如 `20251214`。用于追溯月度请求最终用了哪一景。",
    "selected_image_date": "实际选中的影像日期。日级影像通常为 `YYYYMMDD`；前端可展示给用户核对。",
    "result_url": "结果图片下载 URL。前端可继续 GET 该 URL 获取 PNG。",
    "total": "总数量。用于分页或批量操作统计。",
    "success_count": "批量操作成功数量。",
    "error_count": "批量操作失败数量。",
    "results": "批量结果列表。每项包含 patch、状态、结果 URL 或错误信息。",
    "status": "状态字段。常见值包括 `ok`、`ready`、`training`、`running`、`completed`、`failed`。",
}

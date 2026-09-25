from app.models import PortRef, WorkflowEdge, WorkflowManifest, WorkflowNode
from app.workflow import execution_order, validate_workflow


def make_workflow(edges):
    return WorkflowManifest(
        id="demo",
        entry="a",
        exit="c",
        nodes=[
            WorkflowNode(id="a", component="input"),
            WorkflowNode(id="b", component="noop"),
            WorkflowNode(id="c", component="output"),
        ],
        edges=edges,
    )


def test_workflow_validates_and_orders_nodes():
    workflow = make_workflow(
        [
            WorkflowEdge(from_ref=PortRef(node="a", port="output"), to_ref=PortRef(node="b", port="input")),
            WorkflowEdge(from_ref=PortRef(node="b", port="output"), to_ref=PortRef(node="c", port="input")),
        ]
    )
    result = validate_workflow(workflow)
    assert result.valid is True
    assert execution_order(workflow) == ["a", "b", "c"]


def test_workflow_rejects_cycle():
    workflow = make_workflow(
        [
            WorkflowEdge(from_ref=PortRef(node="a", port="output"), to_ref=PortRef(node="b", port="input")),
            WorkflowEdge(from_ref=PortRef(node="b", port="output"), to_ref=PortRef(node="c", port="input")),
            WorkflowEdge(from_ref=PortRef(node="c", port="output"), to_ref=PortRef(node="a", port="input")),
        ]
    )
    result = validate_workflow(workflow)
    assert result.valid is False
    assert any("cycle" in error for error in result.errors)

import torch
from .eval_vis import calculate_metrics


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for batched_graph, target in val_loader:
            batched_graph = batched_graph.to(device)
            h = batched_graph.ndata['feat'].float().to(device)
            x = batched_graph.ndata['coord'].float().to(device)
            v = torch.zeros_like(x).to(device)
            pairlist = torch.stack(batched_graph.edges(), dim=0).to(device)
            edge_attr = batched_graph.edata['attr'].float().to(device)
            target = target.to(device)

            output = model(batched_graph, h, x, v, pairlist, edge_attr)
            loss = criterion(output.squeeze(), target)
            
            total_loss += loss.item() * target.size(0)
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss, predictions, targets


def test(model, test_loader, criterion, device):
    """Test model"""
    avg_loss, predictions, targets = validate(model, test_loader, criterion, device)
    metrics = calculate_metrics(predictions, targets)
    metrics['loss'] = avg_loss
    return metrics
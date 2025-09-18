# 8. SIMULANDO O RETORNO DO SERVIDOR COM P_ensemble E APLICAÇÃO DE KD

# Suponha que o servidor devolveu as probabilidades médias dos dois modelos no validation set
P_ensemble_val = (probs_val1 + probs_val2) / 2  # [N, num_classes]

# Escolhemos um cliente (ex: model1) para refinar usando KD
client_model = model1
client_model.train()

optimizer = torch.optim.Adam(client_model.parameters(), lr=1e-4)
kd_loss_fn = nn.KLDivLoss(reduction="batchmean")

# Vamos usar um subconjunto do validation loader só para demonstrar
for inputs, targets in tqdm(val_loader, desc="Refinando com KD (cliente)"):
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()

    # Forward do cliente
    outputs = client_model(inputs)
    P_local = F.log_softmax(outputs, dim=1)  # log-probs para KLDivLoss

    # Pegamos o batch correspondente do ensemble (simulado)
    # Atenção: aqui estou simplificando usando P_ensemble_val já calculado,
    # mas em prática isso viria do servidor via MQTT
    batch_indices = torch.arange(len(targets))  # só para alinhar
    P_teacher = P_ensemble_val[batch_indices].to(device)  # já está em probabilidade

    # KL divergence
    loss_kd = kd_loss_fn(P_local, P_teacher)

    # (Opcional) combinar com CE se houver rótulo duro
    ce_loss = F.cross_entropy(outputs, targets)
    alpha = 0.5
    loss = alpha * ce_loss + (1 - alpha) * loss_kd

    loss.backward()
    optimizer.step()
    break  # só 1 batch para demonstração

print(f"✅ Exemplo de atualização local com KLDivLoss concluído (loss={loss.item():.4f})")
import torch
from tqdm import tqdm


def model_train(
        epochs, train_dataloader, valid_dataloader, model,
        optimizer, criterion, batch_size, device
):
    dataloader_dict = {
        "Train": train_dataloader,
        "Valid": valid_dataloader
    }

    # 学習ロスのリスト
    training_data = []
    with tqdm(range(epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description(f"epoch : {epoch + 1}")
            metas = []

            # 学習中orテスト中で処理を分岐
            for phase in ["Train", "Valid"]:
                if phase == "Train":
                    model.train()
                else:
                    model.eval()

                epoch_loss = 0.0

                for inputs, label in dataloader_dict[phase]:
                    inputs = inputs.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()

                    # 学習中のみ、勾配の計算を行う
                    with torch.set_grad_enabled(phase == "Train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, label) ** 0.5

                        # 逆伝播を行う
                        if phase == "Train":
                            loss.backward()
                            optimizer.step()

                        epoch_loss += loss.item() * inputs.size(0)

                # エポックのロスを計算
                epoch_loss /= len(dataloader_dict[phase].dataset) * batch_size

                # エポックごとのロスを連結
                meta = {"Loss": epoch_loss}
                metas.append(meta)

            training_data.append(dict(zip(["Train", "Valid"], metas)))

    return model, training_data

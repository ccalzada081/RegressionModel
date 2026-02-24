import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def main():
    # Carga de datos
    data_path = os.path.join("data", "insurance.csv")
    df = pd.read_csv(data_path)
    df = pd.get_dummies(df, drop_first=True).astype(float)

    # Feature Engineering para darle el empujón extra
    if "smoker_yes" in df.columns:
        df["bmi_smoker"] = df["bmi"] * df["smoker_yes"]

    # El costo sube más rápido a mayor edad (efecto cuadrático)
    df["age2"] = df["age"] ** 2

    X = torch.tensor(df.drop("charges", axis=1).values, dtype=torch.float32)
    Y = torch.tensor(df["charges"].values, dtype=torch.float32).view(-1, 1)

    # Normalización
    mean_X, std_X = X.mean(dim=0), X.std(dim=0)
    X = (X - mean_X) / std_X
    mean_Y, std_Y = Y.mean(), Y.std()
    Y = (Y - mean_Y) / std_Y

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    input_size = X.shape[1]
    output_size = 1

    # Instanciar el modelo
    model = LinearRegression(input_size, output_size)

    learning_rate = 0.01
    n_iters = 5000

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Entrenamiento
    model.train()
    for epoch in range(n_iters):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0 or epoch == n_iters - 1:
            with torch.no_grad():
                y_pred_real = y_pred * std_Y + mean_Y
                Y_train_real = Y_train * std_Y + mean_Y
                real_loss = loss_fn(y_pred_real, Y_train_real).item()
            print(f"epoch {epoch+1:05d}, Train Loss = {real_loss:,.1f}")

    # Evaluación Final
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)

        # Desnormalizar predicciones
        test_pred_real = test_pred * std_Y + mean_Y
        Y_test_real = Y_test * std_Y + mean_Y

        # Calcular el loss en validación para confirmar que cumplimos la meta
        test_loss_real = loss_fn(test_pred_real, Y_test_real).item()

        print("\n" + "=" * 45)
        print(f"TEST LOSS FINAL: {test_loss_real:,.1f}")
        print("=" * 45)

        r2 = r2_score(Y_test_real, test_pred_real)
        print(f"\ntensor({r2.item():.4f})")

        print(f"{'Predicted:':<15} {'real:'}")
        for i in range(6):
            print(
                f"Predicted: {test_pred_real[i].item():10.2f}, real: {Y_test_real[i].item():10.2f}"
            )

    # Creación del reporte CSV (Requisito de la actividad)
    results = pd.DataFrame(
        {
            "Actual Charges": Y_test_real.numpy().flatten(),
            "Predicted Charges": test_pred_real.numpy().flatten(),
        }
    )
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/report.csv", index=False)
    print("\nReporte guardado en results/report.csv!")


if __name__ == "__main__":
    main()

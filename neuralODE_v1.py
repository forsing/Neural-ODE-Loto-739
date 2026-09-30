#!/usr/bin/env python3


"""
Neural ODE — Loto 7/39

Neural ODEs (Neural Ordinary Differential Equations). 

Pokretanje:
    python3 neuralODE_v1.py

Zavisnosti:
    pip install numpy torch

CSV putanja je definisana u CSV_PATH.
Podaci moraju biti poređani od najstarijeg ka najnovijem kolu.
Za obrnut redosled koristi --newest-first.

Izlaz je sedam različitih brojeva od 1 do 39.
Predikcija je procena modela, bez garancije pogodaka.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CSV_PATH = Path(
    "/Users/4c/Desktop/GHQ/data/loto7_4694_k78.csv"
    # "/Users/4c/Desktop/GHQ/data/loto7_4694_k78_loto_2969.csv"
    # "/Users/4c/Desktop/GHQ/data/loto7_4694_k78_loto_plus_1725.csv"
)

SEED = 39
N_NUMBERS = 39
DRAW_SIZE = 7
HIDDEN = 32
RK_STEPS = 4
BATCH = 128


def configure():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)


def read_draws(args):
    path = Path(args.csv)

    if not path.is_file():
        raise FileNotFoundError(f"CSV datoteka ne postoji: {path}")

    text = path.read_text(encoding="utf-8-sig")

    try:
        dialect = csv.Sniffer().sniff(
            text[:8192], delimiters=",;\t"
        )
    except csv.Error:
        dialect = csv.excel

    columns = None
    if args.columns:
        columns = [
            int(value.strip()) - 1
            for value in args.columns.split(",")
        ]
        if (
            len(columns) != DRAW_SIZE
            or len(set(columns)) != DRAW_SIZE
            or min(columns) < 0
        ):
            raise ValueError(
                "--columns mora sadržati 7 različitih "
                "indeksa kolona, počevši od 1."
            )

    result = []
    skip_header = args.header

    for line, row in enumerate(
        csv.reader(text.splitlines(), dialect), 1
    ):
        if not row or all(not cell.strip() for cell in row):
            continue

        if skip_header:
            skip_header = False
            continue

        try:
            if columns is None:
                if len(row) != DRAW_SIZE:
                    raise ValueError(
                        "Očekujem tačno 7 kolona. "
                        "Za dodatne kolone koristi --columns."
                    )
                cells = row
            else:
                cells = [row[index] for index in columns]

            numbers = [int(cell.strip()) for cell in cells]

            if (
                len(set(numbers)) != DRAW_SIZE
                or not all(
                    1 <= number <= N_NUMBERS
                    for number in numbers
                )
            ):
                raise ValueError(
                    "Potrebno je 7 različitih celih "
                    "brojeva od 1 do 39."
                )

            result.append(sorted(numbers))

        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"CSV red {line}: {exc}"
            ) from exc

    if args.newest_first:
        result.reverse()

    minimum = args.window + 40
    if len(result) < minimum:
        raise ValueError(
            f"Potrebno je najmanje {minimum} izvlačenja."
        )

    return np.asarray(result, dtype=np.int64)


class FixedLinear(nn.Module):
    """Analitička inicijalizacija bez slučajnog uzorkovanja."""

    def __init__(self, inputs, outputs, tag):
        super().__init__()

        index = torch.arange(
            outputs * inputs, dtype=torch.float64
        )
        phase = (SEED + tag) * math.sqrt(3.0)

        weight = (
            torch.sin((index + 1) * math.sqrt(2.0) + phase)
            + torch.cos((index + 1) * math.sqrt(5.0) + phase)
        ) / math.sqrt(2.0 * inputs)

        self.weight = nn.Parameter(
            weight.reshape(outputs, inputs)
        )
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class NeuralODE(nn.Module):
    def __init__(self, window):
        super().__init__()

        self.encoder = FixedLinear(
            window * DRAW_SIZE, HIDDEN, 1
        )
        self.f1 = FixedLinear(HIDDEN, HIDDEN, 2)
        self.f2 = FixedLinear(HIDDEN, HIDDEN, 3)
        self.decoder = FixedLinear(
            HIDDEN + DRAW_SIZE + DRAW_SIZE, HIDDEN, 4
        )
        self.output = FixedLinear(HIDDEN, N_NUMBERS, 5)

        values = torch.arange(
            1, N_NUMBERS + 1, dtype=torch.float64
        )
        self.register_buffer("values", values)

        # Za izbor broja v na poziciji p ostaje:
        # C(39 - v, 6 - p) mogućih nastavaka.
        #
        # Kada su naučene korekcije jednake nuli,
        # ova osnova daje jednaku verovatnoću svakoj
        # kompletnoj kombinaciji od sedam brojeva.
        base = torch.zeros(DRAW_SIZE, N_NUMBERS)

        for position in range(DRAW_SIZE):
            remaining = DRAW_SIZE - 1 - position

            for number in range(1, N_NUMBERS + 1):
                available = N_NUMBERS - number

                if available >= remaining:
                    base[position, number - 1] = math.log(
                        math.comb(available, remaining)
                    )

        self.register_buffer("base_logits", base)

        # Početak obuke iz ravnomerne raspodele kombinacija.
        # Ostali slojevi zadržavaju analitičku inicijalizaciju.
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    def dynamics(self, h):
        return 0.5 * torch.tanh(
            self.f2(torch.tanh(self.f1(h)))
        )

    def context(self, x):
        h = torch.tanh(self.encoder(x))
        dt = 1.0 / RK_STEPS

        for _ in range(RK_STEPS):
            k1 = self.dynamics(h)
            k2 = self.dynamics(h + dt * k1 / 2)
            k3 = self.dynamics(h + dt * k2 / 2)
            k4 = self.dynamics(h + dt * k3)

            h = h + dt * (
                k1 + 2 * k2 + 2 * k3 + k4
            ) / 6

        return h

    def logits(self, context, prefix, position):
        marker = context.new_zeros(
            context.shape[0], DRAW_SIZE
        )
        marker[:, position] = 1

        features = torch.cat(
            (
                context,
                prefix / N_NUMBERS,
                marker,
            ),
            dim=1,
        )

        correction = self.output(
            torch.tanh(self.decoder(features))
        )
        logits = (
            correction + self.base_logits[position]
        )

        if position:
            previous = prefix[:, position - 1]
        else:
            previous = context.new_zeros(
                context.shape[0]
            )

        remaining = DRAW_SIZE - 1 - position
        values = self.values[None, :]

        allowed = (
            (values > previous[:, None])
            & (values <= N_NUMBERS - remaining)
        )

        return logits.masked_fill(~allowed, -torch.inf)

    def nll(self, x, target):
        context = self.context(x)
        prefix = x.new_zeros(x.shape[0], DRAW_SIZE)
        losses = []

        for position in range(DRAW_SIZE):
            logits = self.logits(
                context, prefix, position
            )

            losses.append(
                F.cross_entropy(
                    logits,
                    target[:, position] - 1,
                    reduction="none",
                )
            )

            prefix = prefix.clone()
            prefix[:, position] = target[:, position]

        return torch.stack(losses, dim=1).sum(dim=1)


def make_examples(draws, window):
    inputs = np.stack(
        [
            draws[index - window:index].reshape(-1)
            for index in range(window, len(draws))
        ]
    )
    targets = draws[window:]

    return (
        torch.tensor(
            inputs, dtype=torch.float64
        ) / N_NUMBERS,
        torch.tensor(targets, dtype=torch.long),
    )


def make_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )


def train_epoch(model, optimizer, x, y):
    model.train()

    for start in range(0, len(x), BATCH):
        optimizer.zero_grad(set_to_none=True)

        loss = model.nll(
            x[start:start + BATCH],
            y[start:start + BATCH],
        ).mean()

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Obuka je numerički nestabilna."
            )

        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    total = 0.0

    for start in range(0, len(x), BATCH):
        total += model.nll(
            x[start:start + BATCH],
            y[start:start + BATCH],
        ).sum().item()

    return total / len(x)


def choose_epochs(x, y, args):
    split = int(len(x) * 0.8)

    model = NeuralODE(args.window)
    optimizer = make_optimizer(model)

    best_loss = evaluate(
        model, x[split:], y[split:]
    )
    best_epoch = 0
    stale = 0

    print(
        f"Obuka: {split}; "
        f"validacija: {len(x) - split} primera."
    )

    for epoch in range(1, args.epochs + 1):
        train_epoch(
            model, optimizer, x[:split], y[:split]
        )
        validation_loss = evaluate(
            model, x[split:], y[split:]
        )

        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoha {epoch:3d} | "
                f"validacioni NLL: "
                f"{validation_loss:.6f}"
            )

        if stale >= 20:
            break

    uniform_loss = math.log(
        math.comb(N_NUMBERS, DRAW_SIZE)
    )

    print(f"Izabrana epoha: {best_epoch}")
    print(
        f"NLL modela: {best_loss:.6f}; "
        f"ravnomerne raspodele: {uniform_loss:.6f}"
    )
    print(
        "Niži NLL je bolji. Validacija služi "
        "za izbor epohe; nije nezavisan test."
    )

    return best_epoch


@torch.no_grad()
def predict(model, last_input, beam_width):
    model.eval()
    context = model.context(last_input)

    beam = [(0.0, ())]

    for position in range(DRAW_SIZE):
        prefix = context.new_zeros(
            len(beam), DRAW_SIZE
        )

        for row, (_, numbers) in enumerate(beam):
            if numbers:
                prefix[row, :position] = (
                    prefix.new_tensor(numbers)
                )

        log_probabilities = F.log_softmax(
            model.logits(
                context.expand(len(beam), -1),
                prefix,
                position,
            ),
            dim=1,
        ).cpu().numpy()

        candidates = []
        maximum = (
            N_NUMBERS - (DRAW_SIZE - 1 - position)
        )

        for row, (score, numbers) in enumerate(beam):
            minimum = numbers[-1] + 1 if numbers else 1

            for number in range(minimum, maximum + 1):
                candidates.append(
                    (
                        score + float(
                            log_probabilities[
                                row, number - 1
                            ]
                        ),
                        numbers + (number,),
                    )
                )

        candidates.sort(
            key=lambda item: (-item[0], item[1])
        )
        beam = candidates[:beam_width]

    return beam[0][1]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "csv",
        nargs="?",
        default=str(CSV_PATH),
        help="CSV sa istorijom loto izvlačenja",
    )
    parser.add_argument(
        "--header", action="store_true"
    )
    parser.add_argument(
        "--columns",
        help="Sedam indeksa kolona, počevši od 1",
    )
    parser.add_argument(
        "--newest-first", action="store_true"
    )
    parser.add_argument(
        "--window", type=int, default=12
    )
    parser.add_argument(
        "--epochs", type=int, default=120
    )
    parser.add_argument(
        "--beam", type=int, default=512
    )

    args = parser.parse_args()

    if (
        args.window < 1
        or args.epochs < 1
        or args.beam < 1
    ):
        parser.error(
            "--window, --epochs i --beam "
            "moraju biti pozitivni."
        )

    configure()
    draws = read_draws(args)

    print(f"CSV: {Path(args.csv)}")
    print(
        f"Seed: {SEED} | CPU | "
        f"broj izvlačenja: {len(draws)}"
    )
    print(
        "Poslednje učitano izvlačenje:",
        " ".join(f"{n:02d}" for n in draws[-1]),
    )

    x, y = make_examples(draws, args.window)
    epochs = choose_epochs(x, y, args)

    if epochs == 0:
        print(
            "\nModel nije poboljšao ravnomernu "
            "raspodelu na validaciji."
        )
        print(
            "Sve kombinacije imaju jednaku "
            "verovatnoću u izabranoj raspodeli; "
            "nema izdvojene finalne predikcije."
        )
        return

    model = NeuralODE(args.window)
    optimizer = make_optimizer(model)

    print(
        f"Završna obuka na svim podacima: "
        f"{epochs} epoha."
    )

    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, x, y)

        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"Završna epoha: {epoch}/{epochs}"
            )

    last_input = torch.tensor(
        draws[-args.window:].reshape(1, -1),
        dtype=torch.float64,
    ) / N_NUMBERS

    combination = predict(
        model, last_input, args.beam
    )

    if (
        len(combination) != DRAW_SIZE
        or len(set(combination)) != DRAW_SIZE
        or not all(
            1 <= number <= N_NUMBERS
            for number in combination
        )
    ):
        raise RuntimeError(
            "Model je vratio neispravnu kombinaciju."
        )

    print("\nFINALNA PREDIKCIJA:")
    print(
        " ".join(
            f"{number:02d}" for number in combination
        )
    )
    print(
        "Rezultat determinističke beam pretrage; "
        "globalni maksimum nije garantovan."
    )


if __name__ == "__main__":
    main()



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4694_k78.csv
Seed: 39 | CPU | broj izvlačenja: 4694
Poslednje učitano izvlačenje: 08 12 19 20 27 34 38
Obuka: 3745; validacija: 937 primera.
Epoha   1 | validacioni NLL: 16.547681
Epoha  10 | validacioni NLL: 16.560964
Epoha  20 | validacioni NLL: 16.584997
Izabrana epoha: 3
NLL modela: 16.547594; ravnomerne raspodele: 16.548639
Niži NLL je bolji. Validacija služi za izbor epohe; nije nezavisan test.
Završna obuka na svim podacima: 3 epoha.
Završna epoha: 3/3

FINALNA PREDIKCIJA:
07 08 09 10 11 16 25
Rezultat determinističke beam pretrage; globalni maksimum nije garantovan.
"""



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4694_k78_loto_2969.csv
Seed: 39 | CPU | broj izvlačenja: 2969
Poslednje učitano izvlačenje: 01 07 10 21 29 32 36
Obuka: 2365; validacija: 592 primera.
Epoha   1 | validacioni NLL: 16.548944
Epoha  10 | validacioni NLL: 16.563428
Epoha  20 | validacioni NLL: 16.592431
Izabrana epoha: 2
NLL modela: 16.547903; ravnomerne raspodele: 16.548639
Niži NLL je bolji. Validacija služi za izbor epohe; nije nezavisan test.
Završna obuka na svim podacima: 2 epoha.
Završna epoha: 2/2

FINALNA PREDIKCIJA:
03 08 09 10 12 16 22
Rezultat determinističke beam pretrage; globalni maksimum nije garantovan.
"""



"""
CSV: /Users/4c/Desktop/GHQ/data/loto7_4694_k78_loto_plus_1725.csv
Seed: 39 | CPU | broj izvlačenja: 1725
Poslednje učitano izvlačenje: 08 12 19 20 27 34 38
Obuka: 1370; validacija: 343 primera.
Epoha   1 | validacioni NLL: 16.547838
Epoha  10 | validacioni NLL: 16.568249
Epoha  20 | validacioni NLL: 16.636392
Izabrana epoha: 1
NLL modela: 16.547838; ravnomerne raspodele: 16.548639
Niži NLL je bolji. Validacija služi za izbor epohe; nije nezavisan test.
Završna obuka na svim podacima: 1 epoha.
Završna epoha: 1/1

FINALNA PREDIKCIJA:
04 05 06 07 11 12 23
Rezultat determinističke beam pretrage; globalni maksimum nije garantovan.
"""

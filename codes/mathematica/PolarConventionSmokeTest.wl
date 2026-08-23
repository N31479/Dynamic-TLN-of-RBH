ClearAll[lagF, varphi, psi, phiMaster, q, r, lambda, M, e20, m20];

phiDefinition = 2 Sqrt[lagF] *
  (varphi - q psi/(r Sqrt[lambda]));
varphiInverse = phiMaster/(2 Sqrt[lagF]) + q psi/(r Sqrt[lambda]);
inverseCheck = Simplify[
  (varphiInverse /. phiMaster -> phiDefinition) - varphi,
  lagF > 0 && r > 0 && lambda > 0
];

sourceCoefficient = -e20;
responseCoefficient = 2 Sqrt[4 Pi/5] m20;
loveDefinition = -2 Sqrt[4 Pi/5] m20/(M^5 e20);
normalizationCheck = Simplify[
  responseCoefficient/(M^5 sourceCoefficient) - loveDefinition,
  M > 0 && e20 != 0
];

If[TrueQ[inverseCheck == 0] && TrueQ[normalizationCheck == 0],
  Print["Polar reconstruction and Love normalization checks passed."];
  Exit[0],
  Print["Polar convention check failed: ", inverseCheck, " ", normalizationCheck];
  Exit[1]
];

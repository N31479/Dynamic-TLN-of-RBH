                                                      

scriptDirectory = DirectoryName[ExpandFileName[$InputFileName]];
Get[FileNameJoin[{scriptDirectory, "RegularBlackHoleDynamicMaster.wl"}]];

ClearAll[r, ell, M, q, omega];
models = {"Bardeen", "Hayward", "FanWang"};
lambda = (ell - 1) (ell + 2);
n = lambda/2;
reggeWheeler = ell (ell + 1)/r^2 - 6 M/r^3;
zerilli = 2 (n^2 (n + 1) r^3 + 3 n^2 M r^2 + 9 n M^2 r +
    9 M^3)/(r^3 (n r + 3 M)^2);
assumptions = r > 0 && M > 0 && q > 0 && ell >= 2 && Element[ell, Integers];

checks = Table[
  axial = RBHPotentialMatrix[model, "Axial", r, ell, M, q];
  polar = RBHPotentialMatrix[model, "Polar", r, ell, M, q];
  <|
    "MassDerivativeIdentity" -> FullSimplify[
      D[RBHMassFunction[model, r, M, q], r] -
        r^2 RBHLagrangian[model, r, M, q], assumptions],
    "AxialSymmetry" -> FullSimplify[axial[[1, 2]] - axial[[2, 1]], assumptions],
    "PolarSymmetry" -> FullSimplify[polar[[1, 2]] - polar[[2, 1]], assumptions],
    "ReggeWheelerLimit" -> FullSimplify[
      Limit[axial[[1, 1]], q -> 0, Direction -> -1] - reggeWheeler,
      r > 2 M && M > 0 && ell >= 2],
    "ZerilliLimit" -> FullSimplify[
      Limit[polar[[1, 1]], q -> 0, Direction -> -1] - zerilli,
      r > 2 M && M > 0 && ell >= 2],
    "AxialDecouplingLimit" -> FullSimplify[
      Limit[axial[[1, 2]], q -> 0, Direction -> -1],
      r > 2 M && M > 0 && ell >= 2],
    "PolarDecouplingLimit" -> FullSimplify[
      Limit[polar[[1, 2]], q -> 0, Direction -> -1],
      r > 2 M && M > 0 && ell >= 2]
  |>,
  {model, models}
];

checkValues = Flatten[Values /@ checks];
If[! And @@ (PossibleZeroQ /@ checkValues),
  Print["Verification failed: ", checks]; Exit[1]
];

ClearAll[lagFSymbol, varphiSymbol, psiSymbol, phiSymbol, e20, m20];
phiDefinition = 2 Sqrt[lagFSymbol] *
  (varphiSymbol - q psiSymbol/(r Sqrt[lambda]));
varphiInverse = phiSymbol/(2 Sqrt[lagFSymbol]) +
  q psiSymbol/(r Sqrt[lambda]);
polarInverseCheck = FullSimplify[
  (varphiInverse /. phiSymbol -> phiDefinition) - varphiSymbol,
  lagFSymbol > 0 && r > 0 && lambda > 0
];
polarSourceCoefficient = -e20;
polarResponseCoefficient = 2 Sqrt[4 Pi/5] m20;
polarLoveDefinition = -2 Sqrt[4 Pi/5] m20/(M^5 e20);
polarLoveNormalizationCheck = FullSimplify[
  polarResponseCoefficient/(M^5 polarSourceCoefficient) -
    polarLoveDefinition,
  M > 0 && e20 != 0
];
conventionChecks = <|
  "PolarMasterInverse" -> polarInverseCheck,
  "PolarLoveNormalization" -> polarLoveNormalizationCheck
|>;
If[! And @@ (PossibleZeroQ /@ Values[conventionChecks]),
  Print["Convention verification failed: ", conventionChecks]; Exit[1]
];

Export[
  FileNameJoin[{scriptDirectory, "all_models_symbolic_checks.json"}],
  <|
    "Models" -> AssociationThread[
      models,
      Map[AssociationMap[ToString[InputForm[#]] &, #] &, checks]
    ],
    "Conventions" -> AssociationMap[ToString[InputForm[#]] &, conventionChecks]
  |>,
  "JSON"
];
Print["All Bardeen, Hayward, and Fan-Wang symbolic checks passed."];
Exit[0];

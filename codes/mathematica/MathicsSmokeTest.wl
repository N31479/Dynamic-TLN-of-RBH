                                                                         

scriptDirectory = DirectoryName[ExpandFileName[$InputFileName]];
Get[FileNameJoin[{scriptDirectory, "RegularBlackHoleDynamicMaster.wl"}]];

models = {"Bardeen", "Hayward", "FanWang"};
charges = <|"Bardeen" -> 0.4, "Hayward" -> 0.4, "FanWang" -> 0.15|>;
checks = Table[
  axial = N[RBHPotentialMatrix[model, "Axial", 5, 2, 1, charges[model]]];
  polar = N[RBHPotentialMatrix[model, "Polar", 5, 2, 1, charges[model]]];
  equations = RBHRadialEquations[model, "Polar", r, 2, 1, charges[model], omega];
  MatrixQ[axial, NumericQ] && MatrixQ[polar, NumericQ] &&
    Max[Abs[Flatten[axial - Transpose[axial]]]] < 10^-12 &&
    Max[Abs[Flatten[polar - Transpose[polar]]]] < 10^-12 &&
    Length[equations] == 2,
  {model, models}
];

If[And @@ checks,
  Print["Package parse and three-model numerical smoke test passed."];
  Exit[0],
  Print["Smoke test failed: ", checks];
  Exit[1]
];

                                                                                      

scriptDirectory = DirectoryName[ExpandFileName[$InputFileName]];
Get[FileNameJoin[{scriptDirectory, "RegularBlackHoleDynamicMaster.wl"}]];

ClearAll[ScalarPotential, ScalarRadialEquation, psi, r, ell, M, q, omega];

ScalarPotential[model_String, r_, ell_, M_, q_] := Module[{f, fp},
  f = RBHMetric[model, r, M, q];
  fp = D[RBHMetric[model, x, M, q], x] /. x -> r;
  f (ell (ell + 1)/r^2 + fp/r)
];

ScalarRadialEquation[model_String, r_, ell_, M_, q_, omega_] := Module[{f, fp},
  f = RBHMetric[model, r, M, q];
  fp = D[RBHMetric[model, x, M, q], x] /. x -> r;
  f^2 psi''[r] + f fp psi'[r] +
    (omega^2 - ScalarPotential[model, r, ell, M, q]) psi[r] == 0
];

AssociationMap[
  ScalarRadialEquation[#, r, ell, M, q, omega] &,
  {"Bardeen", "Hayward", "FanWang"}
]

                 

BeginPackage["RegularBlackHoleDynamicMaster`"];

RBHMassFunction::usage = "RBHMassFunction[model,r,M,q] gives m(r).";
RBHMetric::usage = "RBHMetric[model,r,M,q] gives f(r)=1-2m(r)/r.";
RBHLagrangian::usage = "RBHLagrangian[model,r,M,q] gives the background NLD Lagrangian.";
RBHLagrangianF::usage = "RBHLagrangianF[model,r,M,q] gives the background derivative L_F.";
RBHKappa::usage = "RBHKappa[model,r,M,q] gives 1+2 F L_FF/L_F.";
RBHPotentialMatrix::usage = "RBHPotentialMatrix[model,parity,r,ell,M,q] gives the symmetric 2 by 2 master potential.";
RBHRadialEquations::usage = "RBHRadialEquations[model,parity,r,ell,M,q,omega] gives the coupled finite-frequency equations.";
RBHPolarMetricReconstruction::usage = "RBHPolarMetricReconstruction[model,r,ell,M,q] gives the polar master-to-metric variables zeta, varphi, K, and H0.";
RBHMasterData::usage = "RBHMasterData[model,r,ell,M,q,omega] returns the background, potentials, and equations.";
Psi::usage = "Psi[r] is the gauge-invariant gravitational master amplitude.";
Phi::usage = "Phi[r] is the gauge-invariant electromagnetic master amplitude.";

Begin["`Private`"];

ClearAll[
  RBHMassFunction, RBHMetric, RBHLagrangian, RBHLagrangianF,
  RBHKappa, RBHPotentialMatrix, RBHRadialEquations, RBHMasterData
  , RBHPolarMetricReconstruction
];

RBHMassFunction[model_String, r_, M_, q_] := Switch[ToLowerCase[model],
  "bardeen", M r^3/(r^2 + q^2)^(3/2),
  "hayward", M r^3/(r^3 + 2 M q^2),
  "fanwang" | "fan-wang" | "fan_wang", M r^3/(r + q)^3,
  _, Message[RBHMassFunction::model, model]; $Failed
];

RBHMassFunction::model = "Unknown model `1`.";

RBHMetric[model_String, r_, M_, q_] :=
  1 - 2 RBHMassFunction[model, r, M, q]/r;

RBHLagrangian[model_String, r_, M_, q_] := Switch[ToLowerCase[model],
  "bardeen", 3 M q^2/(r^2 + q^2)^(5/2),
  "hayward", 6 M^2 q^2/(r^3 + 2 M q^2)^2,
  "fanwang" | "fan-wang" | "fan_wang", 3 M q/(r + q)^4,
  _, Message[RBHLagrangian::model, model]; $Failed
];

RBHLagrangian::model = "Unknown model `1`.";

RBHLagrangianF[model_String, r_, M_, q_] := Switch[ToLowerCase[model],
  "bardeen", 15 M r^6/(2 (r^2 + q^2)^(7/2)),
  "hayward", 18 M^2 r^7/(r^3 + 2 M q^2)^3,
  "fanwang" | "fan-wang" | "fan_wang", 6 M r^5/(q (r + q)^5),
  _, Message[RBHLagrangianF::model, model]; $Failed
];

RBHLagrangianF::model = "Unknown model `1`.";

RBHKappa[model_String, r_, M_, q_] := Module[{logLFPrime},
  logLFPrime = D[Log[RBHLagrangianF[model, x, M, q]], x] /. x -> r;
  1 - r logLFPrime/2
];

RBHPotentialMatrix[
  model_String, parity_String, r_, ell_, M_, q_
] := Module[
  {m, f, fp, lag, lagF, logLFPrime, logLFSecond, kappa, dMinus,
   dPlus, angular, lambda, coupling, a, b, denominator, v11, v22, w},
  m = RBHMassFunction[model, r, M, q];
  f = RBHMetric[model, r, M, q];
  fp = D[RBHMetric[model, x, M, q], x] /. x -> r;
  lag = RBHLagrangian[model, r, M, q];
  lagF = RBHLagrangianF[model, r, M, q];
  logLFPrime = D[Log[RBHLagrangianF[model, x, M, q]], x] /. x -> r;
  logLFSecond = D[Log[RBHLagrangianF[model, x, M, q]], {x, 2}] /. x -> r;
  kappa = 1 - r logLFPrime/2;
  dMinus = -fp logLFPrime/2 + f (-logLFSecond/2 + logLFPrime^2/4);
  dPlus = fp logLFPrime/2 + f (logLFSecond/2 + logLFPrime^2/4);
  angular = ell (ell + 1);
  lambda = (ell - 1) (ell + 2);
  coupling = -q Sqrt[4 lambda lagF]/r^3;

  Switch[ToLowerCase[parity],
    "axial",
      v11 = angular/r^2 - 6 m/r^3 + 2 lag;
      v22 = angular/r^2 + dMinus + 4 q^2 lagF/r^4;
      {{v11, coupling}, {coupling, v22}},
    "polar",
      a = 6 m/r - 2 r^2 lag;
      b = lambda + 4 lagF q^2/r^2;
      denominator = a + lambda;
      v11 = (angular lambda - 2 f lambda + a (a - 4 m/r))/
          (r^2 denominator) + 2 f lambda b/(r^2 denominator^2);
      v22 = kappa angular/r^2 + dPlus +
        4 lagF q^2 (lambda + 1 - f + 2 r^2 lag + 4 f kappa)/
          (r^4 denominator) + 8 f lagF q^2 b/(r^4 denominator^2);
      w = (lambda + 1 - f + 2 r^2 lag + 2 f kappa)/denominator +
        2 f b/denominator^2;
      {{v11, coupling w}, {coupling w, v22}},
    _, Message[RBHPotentialMatrix::parity, parity]; $Failed
  ]
];

RBHPotentialMatrix::parity = "Parity `1` is invalid. Use Polar or Axial.";

RBHRadialEquations[
  model_String, parity_String, r_, ell_, M_, q_, omega_
] := Module[{f, fp, potential, field, residual},
  f = RBHMetric[model, r, M, q];
  fp = D[RBHMetric[model, x, M, q], x] /. x -> r;
  potential = RBHPotentialMatrix[model, parity, r, ell, M, q];
  field = {Psi[r], Phi[r]};
  residual = f^2 D[field, {r, 2}] + f fp D[field, r] +
    (omega^2 IdentityMatrix[2] - f potential).field;
  Thread[residual == {0, 0}]
];

RBHPolarMetricReconstruction[
  model_String, r_, ell_, M_, q_
] := Module[{lag, lagF, m, lambda, angular, a, zeta, varphi, metricK, h0},
  lag = RBHLagrangian[model, r, M, q];
  lagF = RBHLagrangianF[model, r, M, q];
  m = RBHMassFunction[model, r, M, q];
  lambda = (ell - 1) (ell + 2);
  angular = ell (ell + 1);
  a = 6 m/r - 2 r^2 lag;
  zeta = (a + lambda) Psi[r]/Sqrt[lambda];
  varphi = Phi[r]/(2 Sqrt[lagF]) + q Psi[r]/(r Sqrt[lambda]);
  metricK = (8 RBHMetric[model, r, M, q] q lagF varphi/r -
      2 r RBHMetric[model, r, M, q] D[zeta, r] - angular zeta)/
    (r (a + lambda));
  h0 = -D[zeta, r] - r D[metricK, r] + 4 q lagF varphi/r^2;
  <|"Zeta" -> zeta, "Varphi" -> varphi, "K" -> metricK, "H0" -> h0|>
];

RBHMasterData[model_String, r_, ell_, M_, q_, omega_] := <|
  "Model" -> model,
  "MassFunction" -> RBHMassFunction[model, r, M, q],
  "Metric" -> RBHMetric[model, r, M, q],
  "Lagrangian" -> RBHLagrangian[model, r, M, q],
  "LagrangianF" -> RBHLagrangianF[model, r, M, q],
  "Kappa" -> RBHKappa[model, r, M, q],
  "AxialPotential" -> RBHPotentialMatrix[model, "Axial", r, ell, M, q],
  "PolarPotential" -> RBHPotentialMatrix[model, "Polar", r, ell, M, q],
  "AxialEquations" -> RBHRadialEquations[model, "Axial", r, ell, M, q, omega],
  "PolarEquations" -> RBHRadialEquations[model, "Polar", r, ell, M, q, omega]
  , "PolarMetricReconstruction" -> RBHPolarMetricReconstruction[model, r, ell, M, q]
|>;

End[];
EndPackage[];

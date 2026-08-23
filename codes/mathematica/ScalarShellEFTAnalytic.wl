                 

  
                                                                          

                         
                                            
                                                                 
                                                                  
                                         
  

ClearAll["Global`*"];
$IterationLimit = 100000;
$RecursionLimit = 512;

metricX["Bardeen"] := 1 - 2 M x/(1 + q^2 x^2)^(3/2);
metricX["Hayward"] := 1 - 2 M x/(1 + 2 M q^2 x^3);
metricX["FanWang"] := 1 - 2 M x/(1 + q x)^3;

ClearAll[ScalarSourceRecurrence];
ScalarSourceRecurrence[model_String, order_Integer : 7] := Module[
  {metricSeries, b, c, d, n, k, p, previousPower, factor,
   lowerLog, lowerConstant, indicial},

  metricSeries = Normal@Series[metricX[model], {x, 0, order}];
  b = Table[Coefficient[metricSeries, x, k], {k, 0, order}];
  c = ConstantArray[0, order + 1];
  d = ConstantArray[0, order + 1];
  c[[1]] = 1;

  For[n = 1, n <= order, n++,
    p = -3 + n;
    lowerLog = 0;
    lowerConstant = 0;
    For[k = 1, k <= n, k++,
      previousPower = -3 + n - k;
      factor = previousPower (previousPower + 1) +
        k previousPower + k;
      lowerLog += b[[k + 1]] d[[n - k + 1]] factor;
      lowerConstant += b[[k + 1]]
        (c[[n - k + 1]] factor +
         d[[n - k + 1]] (2 previousPower + 1 + k));
    ];
    indicial = p (p + 1) - 6;
    If[TrueQ[Simplify[indicial == 0]],
      d[[n + 1]] = Simplify[-lowerConstant/(2 p + 1)],
      d[[n + 1]] = Simplify[-lowerLog/indicial];
      c[[n + 1]] = Simplify[
        -((2 p + 1) d[[n + 1]] + lowerConstant)/indicial]
    ];
  ];

  <|
    "ConstantCoefficients" -> c,
    "LogCoefficients" -> d,
    "LambdaLog" -> Factor[-d[[6]]]
  |>
];

models = {"Bardeen", "Hayward", "FanWang"};
resultRules = Table[model -> ScalarSourceRecurrence[model],
  {model, models}];
resultFor[model_String] := model /. resultRules;

expectedRules = {
  "Bardeen" -> 3 M q^2 (5 q^2 - 4 M^2)/5,
  "Hayward" -> 32 M^3 q^2/5,
  "FanWang" -> 8 M q^2 (6 M^2 + 20 M q + 15 q^2)/5
};

staticCheckRules = Table[
  model -> TrueQ[Simplify[
    resultFor[model]["LambdaLog"] - (model /. expectedRules)] === 0],
  {model, models}
];
staticChecks = staticCheckRules;

lambdaFinite = Symbol["lambdaFinite"];
mu = Symbol["mu"];
lambdaSymbol = Symbol["lambdaLog"];
dynamicTrial = lambdaSymbol Log[mu r]/6 +
  (lambdaFinite - lambdaSymbol/6)/6;
dynamicCheck = TrueQ[
  Cancel[Together[
    D[dynamicTrial, {r, 2}] - 6 dynamicTrial/r^2 +
    (lambdaSymbol Log[mu r] + lambdaFinite)/r^2
  ]] === 0
];
dynamicCheckRules = Table[model -> dynamicCheck, {model, models}];
dynamicChecks = dynamicCheckRules;

lambdaRules = Table[
  model -> resultFor[model]["LambdaLog"],
  {model, models}
];

sigma = Symbol["sigma"];
staticResponse = lambdaSymbol Log[mu r] + lambdaFinite;
staticScaleCheck = TrueQ[Simplify[
  (staticResponse /. {
      mu -> Exp[sigma] mu,
      lambdaFinite -> lambdaFinite - sigma lambdaSymbol
    }) - staticResponse] === 0];
dynamicScaleCheck = TrueQ[Simplify[
  (dynamicTrial /. {
      mu -> Exp[sigma] mu,
      lambdaFinite -> lambdaFinite - sigma lambdaSymbol
    }) - dynamicTrial] === 0];

lambdaFin[model_String] := Symbol["lambdaFin" <> model];
lambdaTwo[model_String] := Symbol["lambdaTwo" <> model];
schemeTable = AssociationMap[
  Function[model,
    With[{lambda = resultFor[model]["LambdaLog"]},
      <|
        "SchemeIndependentLambdaLog" -> lambda,
        "SchemeIndependentStaticShellLog" ->
          4 Pi lambda Log[mu R]/3,
        "SchemeDependentStaticShellFinite" ->
          4 Pi (5 lambdaFin[model] - lambda)/15,
        "SchemeIndependentOmega2LogCoefficient" -> lambda/6,
        "Omega2ScaleInvariantParticularCombination" ->
          lambda Log[mu r]/6 + (lambdaFin[model] - lambda/6)/6,
        "SchemeDependentOmega2FiniteData" ->
          lambdaFin[model]/6 + lambdaTwo[model]/r^2
      |>
    ]
  ],
  models
];
passed = And @@ Cases[
  Join[staticCheckRules, dynamicCheckRules],
  Rule[_, value_] :> value
] && staticScaleCheck && dynamicScaleCheck;

report = <|
  "LambdaLog" -> lambdaRules,
  "StaticChecks" -> staticChecks,
  "DynamicParticularChecks" -> dynamicChecks,
  "StaticScaleCheck" -> staticScaleCheck,
  "DynamicScaleCheck" -> dynamicScaleCheck,
  "SchemeTable" -> schemeTable,
  "Passed" -> passed
|>;

Print[report];
If[! TrueQ[report["Passed"]], Exit[1]];

function submit()
  addpath('../lib');

  conf.assignmentKey = 'UkTlA-FyRRKV5ooohuwU6A';
  conf.itemName = 'Linear Regression with Multiple Variables';
  conf.partArrays = { ...
    { ...
      'DCRbJ', ...
      { 'warmUpExercise.m' }, ...
      'Warm-up Exercise', ...
    }, ...
    { ...
      'BGa4S', ...
      { 'computeCost.m' }, ...
      'Computing Cost (for One Variable)', ...
    }, ...
    { ...
      'b65eO', ...
      { 'gradientDescent.m' }, ...
      'Gradient Descent (for One Variable)', ...
    }, ...
    { ...
      'BbS8u', ...
      { 'featureNormalize.m' }, ...
      'Feature Normalization', ...
    }, ...
    { ...
      'FBlE2', ...
      { 'computeCostMulti.m' }, ...
      'Computing Cost (for Multiple Variables)', ...
    }, ...
    { ...
      'RZAZC', ...
      { 'gradientDescentMulti.m' }, ...
      'Gradient Descent (for Multiple Variables)', ...
    }, ...
    { ...
      '7m5Eu', ...
      { 'normalEqn.m' }, ...
      'Normal Equations', ...
    }, ...
  };
  conf.output = @output;

  submitWithConfiguration(conf);
end

function out = output(partId)
  % Random Test Cases
  X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))'];
  Y1 = X1(:,2) + sin(X1(:,1)) + cos(X1(:,2));
  X2 = [X1 X1(:,2).^0.5 X1(:,2).^0.25];
  Y2 = Y1.^0.5 + Y1;
  if partId == 'DCRbJ'
    out = sprintf('%0.5f ', warmUpExercise());
  elseif partId == 'BGa4S'
    out = sprintf('%0.5f ', computeCost(X1, Y1, [0.5 -0.5]'));
  elseif partId == 'b65eO'
    out = sprintf('%0.5f ', gradientDescent(X1, Y1, [0.5 -0.5]', 0.01, 10));
  elseif partId == 'BbS8u'
    out = sprintf('%0.5f ', featureNormalize(X2(:,2:4)));
  elseif partId == 'FBlE2'
    out = sprintf('%0.5f ', computeCostMulti(X2, Y2, [0.1 0.2 0.3 0.4]'));
  elseif partId == 'RZAZC'
    out = sprintf('%0.5f ', gradientDescentMulti(X2, Y2, [-0.1 -0.2 -0.3 -0.4]', 0.01, 10));
  elseif partId == '7m5Eu'
    out = sprintf('%0.5f ', normalEqn(X2, Y2));
  end 
end

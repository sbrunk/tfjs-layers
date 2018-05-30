import { scalar, tensor1d, tensor2d, zeros } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import * as losses from './losses';
import { describeMathCPUAndGPU, expectTensorsClose } from './utils/test_utils';
describeMathCPUAndGPU('meanSquaredError', function () {
    it('1D', function () {
        var yTrue = zeros([3]);
        var yPred = tensor1d([1, 2, 3]);
        var expectedVal = scalar((1 * 1 + 2 * 2 + 3 * 3) / 3);
        var result = losses.meanSquaredError(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
    it('2D', function () {
        var yTrue = zeros([2, 2]);
        var yPred = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var expectedVal = tensor1d([(1 * 1 + 2 * 2) / 2, (3 * 3 + 4 * 4) / 2]);
        var result = losses.meanSquaredError(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('meanAbsoluteError', function () {
    it('1D', function () {
        var yTrue = zeros([3]);
        var yPred = tensor1d([-1, -2, -3]);
        var expectedVal = scalar((1 + 2 + 3) / 3);
        var result = losses.meanAbsoluteError(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
    it('2D', function () {
        var yTrue = zeros([2, 2]);
        var yPred = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
        var expectedVal = tensor1d([(1 + 2) / 2, (3 + 4) / 2]);
        var result = losses.meanAbsoluteError(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('meanAbsolutePercentageError', function () {
    it('1D', function () {
        var yTrue = tensor1d([-1, -2, -3]);
        var yPred = zeros([3]);
        var expectedVal = scalar((1 + 2 + 3) / (1 + 2 + 3) * 100);
        var result = losses.meanAbsolutePercentageError(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
    it('2D', function () {
        var yTrue = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
        var yPred = zeros([2, 2]);
        var expectedVal = tensor1d([(1 + 2) / (1 + 2) * 100, (3 + 4) / (3 + 4) * 100]);
        var result = losses.meanAbsolutePercentageError(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('meanSquaredLogarithmicError', function () {
    function meanSquaredLogErrorFor1DArray(x, y) {
        var calcLog = function (val) { return Math.log(Math.max(val, K.epsilon()) + 1); };
        var logX = x.map(calcLog);
        var logY = y.map(calcLog);
        var acc = 0.0;
        for (var i = 0; i < x.length; i++) {
            var diff = logX[i] - logY[i];
            acc += diff * diff;
        }
        return acc / x.length;
    }
    it('2D', function () {
        var yTrue = zeros([2, 2]);
        var yPred = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var expectedVal = tensor1d([
            meanSquaredLogErrorFor1DArray([1, 2], [0, 0]),
            meanSquaredLogErrorFor1DArray([3, 4], [0, 0])
        ]);
        var result = losses.meanSquaredLogarithmicError(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('squaredHinge', function () {
    it('2D', function () {
        var yTrue = tensor2d([[-1, 2], [-3, 2]], [2, 2]);
        var yPred = tensor2d([[-3, 5], [3, -2]], [2, 2]);
        var secondRow = [1 - (-3 * 3), 1 - (2 * -2)].map(function (x) { return x * x; });
        var expectedVal = tensor1d([0, (secondRow[0] + secondRow[1]) / 2]);
        var result = losses.squaredHinge(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('hinge', function () {
    it('2D', function () {
        var yTrue = tensor2d([[-1, 2], [-3, 2]], [2, 2]);
        var yPred = tensor2d([[-3, 5], [3, -2]], [2, 2]);
        var secondRow = [1 - (-3 * 3), 1 - (2 * -2)];
        var expectedVal = tensor1d([0, (secondRow[0] + secondRow[1]) / 2]);
        var result = losses.hinge(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('categoricalHinge', function () {
    it('2D', function () {
        var yTrue = tensor2d([[0, 1, 0], [1, 0, 0]], [2, 3]);
        var yPred = tensor2d([[0, 2, 0], [1, 3, 2]], [2, 3]);
        var secondRowPos = 1 * 1;
        var secondRowNeg = Math.max(1 * 3, 1 * 2);
        var secondRowVal = secondRowNeg - secondRowPos + 1;
        var expectedVal = tensor1d([0, secondRowVal]);
        var result = losses.categoricalHinge(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('logcosh', function () {
    function _logcosh(x) {
        return x + Math.log(Math.exp(-2 * x) + 1) - Math.log(2);
    }
    it('2D', function () {
        var yTrue = zeros([2, 2]);
        var yPred = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var firstRow = [1, 2].map(_logcosh);
        var secondRow = [3, 4].map(_logcosh);
        var firstVal = (firstRow[0] + firstRow[1]) / 2;
        var secondVal = (secondRow[0] + secondRow[1]) / 2;
        var expectedVal = tensor1d([firstVal, secondVal]);
        var result = losses.logcosh(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('categoricalCrossentropy', function () {
    it('2D', function () {
        var yTrue = tensor2d([[1, 0], [0, 1]], [2, 2]);
        var yPred = yTrue;
        var expectedVal = zeros([2]);
        var result = losses.categoricalCrossentropy(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('sparseCategoricalCrossentropy', function () {
    it('2D', function () {
        var yTrue = tensor1d([0, 1]);
        var yPred = tensor2d([[1, 0], [0, 1]], [2, 2]);
        var expectedVal = zeros([2]);
        var result = losses.sparseCategoricalCrossentropy(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('binaryCrossentropy', function () {
    it('2D', function () {
        var yTrue = tensor2d([[1, 0], [1, 0]], [2, 2]);
        var yPred = tensor2d([[1, 2], [20, 10]], [2, 2]);
        var crossEntropy = K.binaryCrossentropy(yTrue, yPred).dataSync();
        var expectedVal = tensor1d([
            (crossEntropy[0] + crossEntropy[1]) / 2,
            (crossEntropy[2] + crossEntropy[3]) / 2
        ]);
        var result = losses.binaryCrossentropy(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('kullbackLeiblerDivergence', function () {
    function klElement(actual, predicted) {
        actual = Math.max(actual, K.epsilon());
        predicted = Math.max(predicted, K.epsilon());
        return actual * Math.log(actual / predicted);
    }
    it('2D', function () {
        var yTrue = tensor2d([[1, 0], [1, 0]], [2, 2]);
        var yPred = tensor2d([[0.25, 0.75], [0.9, 0.1]], [2, 2]);
        var expectedVal = tensor1d([
            klElement(1, 0.25) + klElement(0, 0.75),
            klElement(1, 0.9) + klElement(0, 0.1),
        ]);
        var result = losses.kullbackLeiblerDivergence(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('poisson', function () {
    function poissonElement(actual, predicted) {
        return predicted - actual * Math.log(predicted + K.epsilon());
    }
    it('2D', function () {
        var yTrue = tensor2d([[1, 0], [1, 0]], [2, 2]);
        var yPred = tensor2d([[0.25, 0.75], [0.9, 0.1]], [2, 2]);
        var expectedVal = tensor1d([
            (poissonElement(1, 0.25) + poissonElement(0, 0.75)) / 2,
            (poissonElement(1, 0.9) + poissonElement(0, 0.1)) / 2,
        ]);
        var result = losses.poisson(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describeMathCPUAndGPU('cosineProximity', function () {
    it('2D', function () {
        var z = Math.sqrt(2) / 2;
        var yTrue = tensor2d([[1, 0], [1, 0]], [2, 2]);
        var yPred = tensor2d([[z, z], [0, 1]], [2, 2]);
        var expectedVal = tensor1d([-1 * z, 0]);
        var result = losses.cosineProximity(yTrue, yPred);
        expectTensorsClose(result, expectedVal);
    });
});
describe('losses get', function () {
    var _loop_1 = function (lossName) {
        it("can get " + lossName, function () {
            losses.get(lossName);
        });
    };
    for (var _i = 0, _a = ['meanSquaredError', 'meanAbsoluteError',
        'meanAbsolutePercentageError', 'meanSquaredLogarithmicError',
        'squaredHinge', 'hinge', 'categoricalHinge', 'logcosh',
        'categoricalCrossentropy', 'sparseCategoricalCrossentropy',
        'binaryCrossentropy', 'kullbackLeiblerDivergence', 'poisson',
        'cosineProximity']; _i < _a.length; _i++) {
        var lossName = _a[_i];
        _loop_1(lossName);
    }
    it("get custom loss works", function () {
        var customLoss = function (x, y) { return scalar(42.0); };
        expect(losses.get(customLoss)).toEqual(customLoss);
    });
});
//# sourceMappingURL=losses_test.js.map
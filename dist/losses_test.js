"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var K = require("./backend/tfjs_backend");
var losses = require("./losses");
var test_utils_1 = require("./utils/test_utils");
test_utils_1.describeMathCPUAndGPU('meanSquaredError', function () {
    it('1D', function () {
        var yTrue = tfjs_core_1.zeros([3]);
        var yPred = tfjs_core_1.tensor1d([1, 2, 3]);
        var expectedVal = tfjs_core_1.scalar((1 * 1 + 2 * 2 + 3 * 3) / 3);
        var result = losses.meanSquaredError(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
    it('2D', function () {
        var yTrue = tfjs_core_1.zeros([2, 2]);
        var yPred = tfjs_core_1.tensor2d([[1, 2], [3, 4]], [2, 2]);
        var expectedVal = tfjs_core_1.tensor1d([(1 * 1 + 2 * 2) / 2, (3 * 3 + 4 * 4) / 2]);
        var result = losses.meanSquaredError(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('meanAbsoluteError', function () {
    it('1D', function () {
        var yTrue = tfjs_core_1.zeros([3]);
        var yPred = tfjs_core_1.tensor1d([-1, -2, -3]);
        var expectedVal = tfjs_core_1.scalar((1 + 2 + 3) / 3);
        var result = losses.meanAbsoluteError(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
    it('2D', function () {
        var yTrue = tfjs_core_1.zeros([2, 2]);
        var yPred = tfjs_core_1.tensor2d([[-1, -2], [-3, -4]], [2, 2]);
        var expectedVal = tfjs_core_1.tensor1d([(1 + 2) / 2, (3 + 4) / 2]);
        var result = losses.meanAbsoluteError(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('meanAbsolutePercentageError', function () {
    it('1D', function () {
        var yTrue = tfjs_core_1.tensor1d([-1, -2, -3]);
        var yPred = tfjs_core_1.zeros([3]);
        var expectedVal = tfjs_core_1.scalar((1 + 2 + 3) / (1 + 2 + 3) * 100);
        var result = losses.meanAbsolutePercentageError(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor2d([[-1, -2], [-3, -4]], [2, 2]);
        var yPred = tfjs_core_1.zeros([2, 2]);
        var expectedVal = tfjs_core_1.tensor1d([(1 + 2) / (1 + 2) * 100, (3 + 4) / (3 + 4) * 100]);
        var result = losses.meanAbsolutePercentageError(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('meanSquaredLogarithmicError', function () {
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
        var yTrue = tfjs_core_1.zeros([2, 2]);
        var yPred = tfjs_core_1.tensor2d([[1, 2], [3, 4]], [2, 2]);
        var expectedVal = tfjs_core_1.tensor1d([
            meanSquaredLogErrorFor1DArray([1, 2], [0, 0]),
            meanSquaredLogErrorFor1DArray([3, 4], [0, 0])
        ]);
        var result = losses.meanSquaredLogarithmicError(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('squaredHinge', function () {
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor2d([[-1, 2], [-3, 2]], [2, 2]);
        var yPred = tfjs_core_1.tensor2d([[-3, 5], [3, -2]], [2, 2]);
        var secondRow = [1 - (-3 * 3), 1 - (2 * -2)].map(function (x) { return x * x; });
        var expectedVal = tfjs_core_1.tensor1d([0, (secondRow[0] + secondRow[1]) / 2]);
        var result = losses.squaredHinge(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('hinge', function () {
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor2d([[-1, 2], [-3, 2]], [2, 2]);
        var yPred = tfjs_core_1.tensor2d([[-3, 5], [3, -2]], [2, 2]);
        var secondRow = [1 - (-3 * 3), 1 - (2 * -2)];
        var expectedVal = tfjs_core_1.tensor1d([0, (secondRow[0] + secondRow[1]) / 2]);
        var result = losses.hinge(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('categoricalHinge', function () {
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor2d([[0, 1, 0], [1, 0, 0]], [2, 3]);
        var yPred = tfjs_core_1.tensor2d([[0, 2, 0], [1, 3, 2]], [2, 3]);
        var secondRowPos = 1 * 1;
        var secondRowNeg = Math.max(1 * 3, 1 * 2);
        var secondRowVal = secondRowNeg - secondRowPos + 1;
        var expectedVal = tfjs_core_1.tensor1d([0, secondRowVal]);
        var result = losses.categoricalHinge(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('logcosh', function () {
    function _logcosh(x) {
        return x + Math.log(Math.exp(-2 * x) + 1) - Math.log(2);
    }
    it('2D', function () {
        var yTrue = tfjs_core_1.zeros([2, 2]);
        var yPred = tfjs_core_1.tensor2d([[1, 2], [3, 4]], [2, 2]);
        var firstRow = [1, 2].map(_logcosh);
        var secondRow = [3, 4].map(_logcosh);
        var firstVal = (firstRow[0] + firstRow[1]) / 2;
        var secondVal = (secondRow[0] + secondRow[1]) / 2;
        var expectedVal = tfjs_core_1.tensor1d([firstVal, secondVal]);
        var result = losses.logcosh(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('categoricalCrossentropy', function () {
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor2d([[1, 0], [0, 1]], [2, 2]);
        var yPred = yTrue;
        var expectedVal = tfjs_core_1.zeros([2]);
        var result = losses.categoricalCrossentropy(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('sparseCategoricalCrossentropy', function () {
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor1d([0, 1]);
        var yPred = tfjs_core_1.tensor2d([[1, 0], [0, 1]], [2, 2]);
        var expectedVal = tfjs_core_1.zeros([2]);
        var result = losses.sparseCategoricalCrossentropy(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('binaryCrossentropy', function () {
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor2d([[1, 0], [1, 0]], [2, 2]);
        var yPred = tfjs_core_1.tensor2d([[1, 2], [20, 10]], [2, 2]);
        var crossEntropy = K.binaryCrossentropy(yTrue, yPred).dataSync();
        var expectedVal = tfjs_core_1.tensor1d([
            (crossEntropy[0] + crossEntropy[1]) / 2,
            (crossEntropy[2] + crossEntropy[3]) / 2
        ]);
        var result = losses.binaryCrossentropy(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('kullbackLeiblerDivergence', function () {
    function klElement(actual, predicted) {
        actual = Math.max(actual, K.epsilon());
        predicted = Math.max(predicted, K.epsilon());
        return actual * Math.log(actual / predicted);
    }
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor2d([[1, 0], [1, 0]], [2, 2]);
        var yPred = tfjs_core_1.tensor2d([[0.25, 0.75], [0.9, 0.1]], [2, 2]);
        var expectedVal = tfjs_core_1.tensor1d([
            klElement(1, 0.25) + klElement(0, 0.75),
            klElement(1, 0.9) + klElement(0, 0.1),
        ]);
        var result = losses.kullbackLeiblerDivergence(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('poisson', function () {
    function poissonElement(actual, predicted) {
        return predicted - actual * Math.log(predicted + K.epsilon());
    }
    it('2D', function () {
        var yTrue = tfjs_core_1.tensor2d([[1, 0], [1, 0]], [2, 2]);
        var yPred = tfjs_core_1.tensor2d([[0.25, 0.75], [0.9, 0.1]], [2, 2]);
        var expectedVal = tfjs_core_1.tensor1d([
            (poissonElement(1, 0.25) + poissonElement(0, 0.75)) / 2,
            (poissonElement(1, 0.9) + poissonElement(0, 0.1)) / 2,
        ]);
        var result = losses.poisson(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
    });
});
test_utils_1.describeMathCPUAndGPU('cosineProximity', function () {
    it('2D', function () {
        var z = Math.sqrt(2) / 2;
        var yTrue = tfjs_core_1.tensor2d([[1, 0], [1, 0]], [2, 2]);
        var yPred = tfjs_core_1.tensor2d([[z, z], [0, 1]], [2, 2]);
        var expectedVal = tfjs_core_1.tensor1d([-1 * z, 0]);
        var result = losses.cosineProximity(yTrue, yPred);
        test_utils_1.expectTensorsClose(result, expectedVal);
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
        var customLoss = function (x, y) { return tfjs_core_1.scalar(42.0); };
        expect(losses.get(customLoss)).toEqual(customLoss);
    });
});
//# sourceMappingURL=losses_test.js.map
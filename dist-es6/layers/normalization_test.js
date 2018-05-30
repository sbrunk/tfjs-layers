var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
import { onesLike, tensor2d, tensor3d, tensor4d, train, zeros, zerosLike } from '@tensorflow/tfjs-core';
import * as tfl from '../index';
import { SymbolicTensor } from '../types';
import { describeMathCPU, describeMathCPUAndGPU, expectTensorsClose } from '../utils/test_utils';
describeMathCPU('BatchNormalization Layers: Symbolic', function () {
    var validInputShapes = [[4, 6], [2, 3, 4], [2, 3, 4, 5]];
    var _loop_1 = function (inputShape) {
        var testTitle = "shape=" + JSON.stringify(inputShape);
        it(testTitle, function () {
            var x = new SymbolicTensor('float32', inputShape, null, [], null);
            var layer = tfl.layers.batchNormalization({});
            var y = layer.apply(x);
            expect(y.dtype).toEqual(x.dtype);
            expect(y.shape).toEqual(x.shape);
        });
    };
    for (var _i = 0, validInputShapes_1 = validInputShapes; _i < validInputShapes_1.length; _i++) {
        var inputShape = validInputShapes_1[_i];
        _loop_1(inputShape);
    }
    it('Undetermined dim axis leads to ValueError', function () {
        var x = new SymbolicTensor('float32', [null, 2, 3], null, [], null);
        var layer = tfl.layers.batchNormalization({ axis: 0 });
        expect(function () { return layer.apply(x); })
            .toThrowError(/Axis 0 of input tensor should have a defined dimension.*/);
    });
});
describeMathCPUAndGPU('BatchNormalization Layers: Tensor', function () {
    var dimensions = [2, 3, 4];
    var axisValues = [0, -1];
    var _loop_2 = function (dim) {
        var _loop_3 = function (axis) {
            var testTitle = "Inference, " + dim + "D, axis=" + axis;
            it(testTitle, function () {
                var layer = tfl.layers.batchNormalization({ axis: axis });
                var x;
                if (dim === 2) {
                    x = tensor2d([[1, 2], [3, 4]], [2, 2]);
                }
                else if (dim === 3) {
                    x = tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]], [2, 2, 2]);
                }
                else if (dim === 4) {
                    x = tensor4d([
                        [[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]],
                        [[[-1, -2], [-3, -4]], [[1, 2], [3, 4]]]
                    ], [2, 2, 2, 2]);
                }
                var y = layer.apply(x, { training: false });
                expectTensorsClose(y, x, 0.01);
            });
        };
        for (var _i = 0, axisValues_1 = axisValues; _i < axisValues_1.length; _i++) {
            var axis = axisValues_1[_i];
            _loop_3(axis);
        }
    };
    for (var _i = 0, dimensions_1 = dimensions; _i < dimensions_1.length; _i++) {
        var dim = dimensions_1[_i];
        _loop_2(dim);
    }
    it('no center', function () {
        var layer = tfl.layers.batchNormalization({ center: false, axis: 0 });
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        expectTensorsClose(layer.apply(x), x, 0.01);
        expect(layer.getWeights().length).toEqual(3);
        expectTensorsClose(layer.getWeights()[0], onesLike(layer.getWeights()[0]));
        expectTensorsClose(layer.getWeights()[1], zerosLike(layer.getWeights()[1]));
        expectTensorsClose(layer.getWeights()[2], onesLike(layer.getWeights()[2]));
    });
    it('no scale', function () {
        var layer = tfl.layers.batchNormalization({ scale: false, axis: 0 });
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        expectTensorsClose(layer.apply(x), x, 0.01);
        expect(layer.getWeights().length).toEqual(3);
        expectTensorsClose(layer.getWeights()[0], zerosLike(layer.getWeights()[0]));
        expectTensorsClose(layer.getWeights()[1], zerosLike(layer.getWeights()[1]));
        expectTensorsClose(layer.getWeights()[2], onesLike(layer.getWeights()[2]));
    });
    it('no center, no scale', function () {
        var layer = tfl.layers.batchNormalization({ scale: false, center: false });
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        expectTensorsClose(layer.apply(x), x, 0.01);
        expect(layer.getWeights().length).toEqual(2);
        expectTensorsClose(layer.getWeights()[0], zerosLike(layer.getWeights()[0]));
        expectTensorsClose(layer.getWeights()[1], onesLike(layer.getWeights()[1]));
    });
    it('Fit: 2D, BatchNorm Layer Only', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var layer1, model, xs1, ys;
        return __generator(this, function (_a) {
            layer1 = tfl.layers.batchNormalization({ inputShape: [4] });
            model = tfl.sequential({ layers: [layer1] });
            model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
            xs1 = tensor2d([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], [3, 4]);
            ys = zeros([3, 4]);
            model.fit(xs1, ys, { epochs: 2, batchSize: 3 })
                .then(function (history) {
                expect(history.history['loss'][0]).toBeCloseTo(0.9998891353607178);
                expect(history.history['loss'][1]).toBeCloseTo(0.9899163246154785);
                var gammaValue = layer1.getWeights()[0];
                expectTensorsClose(gammaValue, [0.9900254, 0.9900257, 0.9900262, 0.9900271]);
                var betaValue = layer1.getWeights()[1];
                expectTensorsClose(betaValue, [2.9802322e-10, 1.4901161e-10, 8.9406960e-10, -7.4505802e-10]);
                var movingMeanValue = layer1.getWeights()[2];
                expectTensorsClose(movingMeanValue, [5.0000086, 5.6666765, 6.333345, 7.000012]);
                var movingVarianceValue = layer1.getWeights()[3];
                expectTensorsClose(movingVarianceValue, [37.018574, 22.344547, 12.339525, 7.003515]);
                done();
            })
                .catch(function (err) {
                console.error(err.stack);
            });
            return [2];
        });
    }); });
    it('Fit: 2D, BatchNorm Layer between two Dense Layers', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var layer1, layer2, layer3, model, optimizer, xs1, ys;
        return __generator(this, function (_a) {
            layer1 = tfl.layers.dense({ units: 4, kernelInitializer: 'ones', useBias: false, inputShape: [4] });
            layer2 = tfl.layers.batchNormalization({ inputShape: [4] });
            layer3 = tfl.layers.dense({ units: 1, kernelInitializer: 'ones', useBias: false });
            model = tfl.sequential({ layers: [layer1, layer2, layer3] });
            optimizer = train.sgd(0.1);
            model.compile({ loss: 'meanSquaredError', optimizer: optimizer });
            xs1 = tensor2d([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], [3, 4]);
            ys = zeros([3, 1]);
            model.fit(xs1, ys, { epochs: 3, batchSize: 3 })
                .then(function (history) {
                expect(history.history['loss'][0]).toBeCloseTo(15.999907493591309);
                expect(history.history['loss'][1]).toBeCloseTo(0.025602197274565697);
                expect(history.history['loss'][2]).toBeCloseTo(0.022478966042399406);
                var dense1KernelValue = layer1.getWeights()[0];
                expectTensorsClose(dense1KernelValue, tensor2d([
                    [0.99999833, 0.99999833, 0.99999833, 0.99999833],
                    [0.9999987, 0.9999987, 0.9999987, 0.9999987],
                    [0.999999, 0.999999, 0.999999, 0.999999],
                    [0.99999934, 0.99999934, 0.99999934, 0.99999934]
                ], [4, 4]));
                var gammaValue = layer2.getWeights()[0];
                expectTensorsClose(gammaValue, [0.18779878, 0.18779878, 0.18779878, 0.18779878]);
                var betaValue = layer2.getWeights()[1];
                expectTensorsClose(betaValue, [5.5367128e-08, 5.5367128e-08, 5.5367128e-08, 5.5367128e-08]);
                var movingMeanValue = layer2.getWeights()[2];
                expectTensorsClose(movingMeanValue, [23.999907, 23.999907, 23.999907, 23.999907]);
                var movingVarianceValue = layer2.getWeights()[3];
                expectTensorsClose(movingVarianceValue, [268.13364, 268.13364, 268.13364, 268.13364]);
                var dense2KernelValue = layer3.getWeights()[0];
                expectTensorsClose(dense2KernelValue, tensor2d([[0.18779878], [0.18779878], [0.18779878], [0.18779878]], [4, 1]));
                done();
            })
                .catch(function (err) {
                console.error(err.stack);
            });
            return [2];
        });
    }); });
    it('Fit: 3D, BatchNorm Layer Only', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var layer1, model, xs1, ys;
        return __generator(this, function (_a) {
            layer1 = tfl.layers.batchNormalization({ inputShape: [2, 2] });
            model = tfl.sequential({ layers: [layer1] });
            model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
            xs1 = tensor3d([[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
            ys = zeros([3, 2, 2]);
            model.fit(xs1, ys, { epochs: 2, batchSize: 3 })
                .then(function (history) {
                expect(history.history['loss'][0]).toBeCloseTo(0.9999215006828308);
                expect(history.history['loss'][1]).toBeCloseTo(0.980024516582489);
                var gammaValue = layer1.getWeights()[0];
                expectTensorsClose(gammaValue, [0.98010117, 0.98010194]);
                var betaValue = layer1.getWeights()[1];
                expectTensorsClose(betaValue, [-1.1175870e-09, 8.1956386e-10]);
                var movingMeanValue = layer1.getWeights()[2];
                expectTensorsClose(movingMeanValue, [5.6666765, 6.333345]);
                var movingVarianceValue = layer1.getWeights()[3];
                expectTensorsClose(movingVarianceValue, [20.270758, 12.269142]);
                done();
            })
                .catch(function (err) {
                console.error(err.stack);
            });
            return [2];
        });
    }); });
});
//# sourceMappingURL=normalization_test.js.map
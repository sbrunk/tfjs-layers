import * as tfc from '@tensorflow/tfjs-core';
import { scalar, memory, tensor1d, tensor2d, tensor3d, tensor4d, zeros } from '@tensorflow/tfjs-core';
import { SymbolicTensor } from '../types';
import { LayerVariable } from '../variables';
import { unique } from '../utils/generic_utils';
import { range } from '../utils/math_utils';
import { describeMathCPU, describeMathCPUAndGPU, expectTensorsClose, expectNoLeakedTensors } from '../utils/test_utils';
import * as K from './tfjs_backend';
describe('TensorMath', function () {
    it('Setting and getting backend', function () {
        var originalBackend = K.getBackend();
        expect(originalBackend).toEqual('webgl');
        K.setBackend('cpu');
        expect(K.getBackend()).toEqual('cpu');
    });
});
describe('shape', function () {
    it('Scalar', function () {
        var x = zeros([]);
        expect(K.shape(x)).toEqual([]);
    });
    it('Tensor1D', function () {
        var x = zeros([3]);
        expect(K.shape(x)).toEqual([3]);
    });
    it('Tensor2D', function () {
        var x = zeros([3, 2]);
        expect(K.shape(x)).toEqual([3, 2]);
    });
    it('Tensor3D', function () {
        var x = zeros([4, 3, 2]);
        expect(K.shape(x)).toEqual([4, 3, 2]);
    });
    it('Tensor4D', function () {
        var x = zeros([4, 3, 2, 1]);
        expect(K.shape(x)).toEqual([4, 3, 2, 1]);
    });
});
describe('intShape', function () {
    it('Scalar', function () {
        var x = zeros([]);
        expect(K.intShape(x)).toEqual([]);
    });
    it('Tensor1D', function () {
        var x = zeros([3]);
        expect(K.intShape(x)).toEqual([3]);
    });
    it('Tensor2D', function () {
        var x = zeros([3, 2]);
        expect(K.intShape(x)).toEqual([3, 2]);
    });
    it('Tensor3D', function () {
        var x = zeros([4, 3, 2]);
        expect(K.intShape(x)).toEqual([4, 3, 2]);
    });
    it('Tensor4D', function () {
        var x = zeros([4, 3, 2, 1]);
        expect(K.intShape(x)).toEqual([4, 3, 2, 1]);
    });
});
describe('ndim', function () {
    it('Scalar', function () {
        var x = zeros([]);
        expect(K.ndim(x)).toEqual(0);
    });
    it('Tensor1D', function () {
        var x = zeros([3]);
        expect(K.ndim(x)).toEqual(1);
    });
    it('Tensor2D', function () {
        var x = zeros([3, 2]);
        expect(K.ndim(x)).toEqual(2);
    });
    it('Tensor3D', function () {
        var x = zeros([4, 3, 2]);
        expect(K.ndim(x)).toEqual(3);
    });
    it('Tensor4D', function () {
        var x = zeros([4, 3, 2, 1]);
        expect(K.ndim(x)).toEqual(4);
    });
});
describe('dtype', function () {
    it('returns float32 for an Tensor', function () {
        var x = zeros([1]);
        expect(K.dtype(x)).toEqual('float32');
    });
    it('returns float32 for a SymbolicTensor', function () {
        var x = new SymbolicTensor('float32', [1], null, [], {});
        expect(K.dtype(x)).toEqual('float32');
    });
});
describeMathCPU('countParams', function () {
    it('Scalar', function () {
        var x = zeros([]);
        expect(K.countParams(x)).toEqual(1);
        expect(K.countParams(new LayerVariable(x).read())).toEqual(1);
    });
    it('Tensor1D', function () {
        var x = zeros([3]);
        expect(K.countParams(x)).toEqual(3);
        expect(K.countParams(new LayerVariable(x).read())).toEqual(3);
    });
    it('Tensor2D', function () {
        var x = zeros([3, 2]);
        expect(K.countParams(x)).toEqual(6);
        expect(K.countParams(new LayerVariable(x).read())).toEqual(6);
    });
    it('Tensor3D', function () {
        var x = zeros([4, 3, 2]);
        expect(K.countParams(x)).toEqual(24);
        expect(K.countParams(new LayerVariable(x).read())).toEqual(24);
    });
    it('Tensor4D', function () {
        var x = zeros([4, 3, 2, 1]);
        expect(K.countParams(x)).toEqual(24);
        expect(K.countParams(new LayerVariable(x).read())).toEqual(24);
    });
});
describeMathCPUAndGPU('cast', function () {
    it('float32 to int32', function () {
        var x = tensor2d([[-1.1, -1.6], [1.1, 2.2], [3.6, 4.7]], [3, 2], 'float32');
        var y = K.cast(x, 'int32');
        expect(y.dtype).toEqual('int32');
        expect(y.shape).toEqual([3, 2]);
        expect(Array.from(y.dataSync())).toEqual([-1, -1, 1, 2, 3, 4]);
    });
    it('int32 to float32', function () {
        var x = tensor2d([[-1, -1], [1, 2], [3, 4]], [3, 2], 'int32');
        var y = K.cast(x, 'float32');
        expect(y.dtype).toEqual('float32');
        expect(y.shape).toEqual([3, 2]);
        expect(Array.from(y.dataSync())).toEqual([-1, -1, 1, 2, 3, 4]);
    });
    it('float32 to bool', function () {
        var x = tensor2d([[-1.1, -1.6], [0.0, 2.2], [3.6, 4.7]], [3, 2], 'float32');
        var y = K.cast(x, 'bool');
        expect(y.dtype).toEqual('bool');
        expect(y.shape).toEqual([3, 2]);
        expect(Array.from(y.dataSync())).toEqual([1, 1, 0, 1, 1, 1]);
    });
    it('bool to float32', function () {
        var x = tensor2d([[0, 1], [0, 1], [1, 0]], [3, 2], 'bool');
        var y = K.cast(x, 'float32');
        expect(y.dtype).toEqual('float32');
        expect(y.shape).toEqual([3, 2]);
        expect(Array.from(y.dataSync())).toEqual([0, 1, 0, 1, 1, 0]);
    });
    it('int32 to bool', function () {
        var x = tensor2d([[-1, -2], [0, 2], [3, 4]], [3, 2], 'int32');
        var y = K.cast(x, 'bool');
        expect(y.dtype).toEqual('bool');
        expect(y.shape).toEqual([3, 2]);
        expect(Array.from(y.dataSync())).toEqual([1, 1, 0, 1, 1, 1]);
    });
    it('bool to int32', function () {
        var x = tensor2d([[0, 1], [0, 1], [1, 0]], [3, 2], 'bool');
        var y = K.cast(x, 'int32');
        expect(y.dtype).toEqual('int32');
        expect(y.shape).toEqual([3, 2]);
        expect(Array.from(y.dataSync())).toEqual([0, 1, 0, 1, 1, 0]);
    });
});
describeMathCPUAndGPU('expandDims', function () {
    it('Scalar to 1D', function () {
        var x = scalar(10);
        expectTensorsClose(K.expandDims(x), tensor1d([10]));
    });
    it('1D to 2D: Last dimension', function () {
        var x = tensor1d([10, 20, 30]);
        expectTensorsClose(K.expandDims(x), tensor2d([[10], [20], [30]], [3, 1]));
    });
    it('1D to 2D: First dimension', function () {
        var x = tensor1d([10, 20, 30]);
        expectTensorsClose(K.expandDims(x, 0), tensor2d([[10, 20, 30]], [1, 3]));
    });
    it('2D to 3D: Last dimension', function () {
        var x = tensor2d([[10, 20], [30, 40]], [2, 2]);
        expectTensorsClose(K.expandDims(x), tensor3d([[[10], [20]], [[30], [40]]], [2, 2, 1]));
    });
    it('2D to 3D: Second dimension', function () {
        var x = tensor2d([[10, 20], [30, 40]], [2, 2]);
        expectTensorsClose(K.expandDims(x, 1), tensor3d([[[10, 20]], [[30, 40]]], [2, 1, 2]));
    });
    it('2D to 3D: First dimension', function () {
        var x = tensor2d([[10, 20], [30, 40]], [2, 2]);
        expectTensorsClose(K.expandDims(x, 0), tensor3d([[[10, 20], [30, 40]]], [1, 2, 2]));
    });
});
describeMathCPUAndGPU('Repeat', function () {
    it('2D array', function () {
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var y = K.repeat(x, 3);
        expectTensorsClose(y, tensor3d([[[1, 2], [1, 2], [1, 2]], [[3, 4], [3, 4], [3, 4]]], [2, 3, 2]));
    });
    it('Non-2D array leads to AssertionError', function () {
        var x = tensor1d([1, 2, 3]);
        expect(function () { return K.repeat(x, 2); })
            .toThrowError(/repeat\(\) expects a rank-2 tensor, but received a rank-1 tensor/);
    });
});
describeMathCPUAndGPU('Flatten', function () {
    it('1D Tensor', function () {
        var x = tensor1d([1, 3, 3, 7]);
        var flattend = K.flatten(x);
        expect(flattend.shape).toEqual([4]);
        expect(flattend.dataSync()).toEqual(new Float32Array([1, 3, 3, 7]));
    });
    it('2D Tensor', function () {
        var x = tensor2d([1, 3, 3, 7], [2, 2]);
        var flattend = K.flatten(x);
        expect(flattend.shape).toEqual([4]);
        expect(flattend.dataSync()).toEqual(new Float32Array([1, 3, 3, 7]));
    });
    it('3D Tensor', function () {
        var x = tensor3d([[[10, 20, 30], [40, 50, 60]], [[-10, -20, -30], [-40, -50, -60]]], [2, 2, 3]);
        var flattend = K.flatten(x);
        expect(flattend.shape).toEqual([12]);
        expect(flattend.dataSync()).toEqual(new Float32Array([
            10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60
        ]));
    });
    it('4D Tensor', function () {
        var x = tensor4d([1, 2, 3, 4, 5, 6, 7, 8, -8, -7, -6, -5, -4, -3, -2, -1], [2, 2, 2, 2]);
        var flattend = K.flatten(x);
        expect(flattend.shape).toEqual([16]);
        expect(flattend.dataSync()).toEqual(new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, -8, -7, -6, -5, -4, -3, -2, -1
        ]));
    });
});
describeMathCPUAndGPU('batchFlatten', function () {
    it('Scalar Tensor leads to error', function () {
        var x = scalar(1337);
        expect(function () { return K.batchFlatten(x); })
            .toThrowError(/batchFlatten requires a minimum rank of 2\. Got rank: 0/);
    });
    it('1D Tensor leads to error', function () {
        var x = tensor1d([1, 3, 3, 7]);
        expect(function () { return K.batchFlatten(x); })
            .toThrowError(/batchFlatten requires a minimum rank of 2\. Got rank: 1/);
    });
    it('2D Tensor', function () {
        var x = tensor2d([1, 3, 3, 7], [2, 2]);
        var batchFlattened = K.batchFlatten(x);
        expect(batchFlattened.shape).toEqual([2, 2]);
        expect(batchFlattened.dataSync()).toEqual(new Float32Array([1, 3, 3, 7]));
    });
    it('3D Tensor', function () {
        var x = tensor3d([[[10, 20, 30], [40, 50, 60]], [[-10, -20, -30], [-40, -50, -60]]], [2, 2, 3]);
        var batchFlattened = K.batchFlatten(x);
        expect(batchFlattened.shape).toEqual([2, 6]);
        expect(batchFlattened.dataSync()).toEqual(new Float32Array([
            10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60
        ]));
    });
    it('4D Tensor', function () {
        var x = tensor4d([1, 2, 3, 4, 5, 6, 7, 8, -8, -7, -6, -5, -4, -3, -2, -1], [2, 2, 2, 2]);
        var batchFlattened = K.batchFlatten(x);
        expect(batchFlattened.shape).toEqual([2, 8]);
        expect(batchFlattened.dataSync()).toEqual(new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, -8, -7, -6, -5, -4, -3, -2, -1
        ]));
    });
});
describeMathCPUAndGPU('sliceAlongFirstAxis', function () {
    var array1DData = [10, 20, 30, 40];
    it('1D', function () {
        var x = tensor1d(array1DData);
        expectTensorsClose(K.sliceAlongFirstAxis(x, 1, 2), tensor1d([20, 30]));
    });
    var array2DData = [[10, 11], [20, 21], [30, 31], [40, 41]];
    it('2D', function () {
        var x = tensor2d(array2DData, [4, 2]);
        expectTensorsClose(K.sliceAlongFirstAxis(x, 1, 2), tensor2d([[20, 21], [30, 31]], [2, 2]));
    });
    var array3DData = [[[10]], [[20]], [[30]], [[40]]];
    it('3D', function () {
        var x = tensor3d(array3DData, [4, 1, 1]);
        expectTensorsClose(K.sliceAlongFirstAxis(x, 1, 2), tensor3d([[[20]], [[30]]], [2, 1, 1]));
    });
    var array4DData = [[[[10]]], [[[20]]], [[[30]]], [[[40]]]];
    it('4D', function () {
        var x = tensor4d(array4DData, [4, 1, 1, 1]);
        expectTensorsClose(K.sliceAlongFirstAxis(x, 1, 2), tensor4d([[[[20]]], [[[30]]]], [2, 1, 1, 1]));
    });
    it('Scalar leads to error', function () {
        expect(function () {
            K.sliceAlongFirstAxis(scalar(24), 0, 1);
        }).toThrow();
    });
});
describeMathCPUAndGPU('sliceAlongLastAxis', function () {
    var array1DData = [10, 20, 30, 40];
    it('1D', function () {
        var x = tensor1d(array1DData);
        expectTensorsClose(K.sliceAlongLastAxis(x, 1, 2), tensor1d([20, 30]));
    });
    var array2DData = [[10, 11, 12, 13], [20, 21, 22, 23]];
    it('2D', function () {
        var x = tensor2d(array2DData, [2, 4]);
        expectTensorsClose(K.sliceAlongLastAxis(x, 1, 2), tensor2d([[11, 12], [21, 22]], [2, 2]));
    });
    var array3DData = [[[10, 20, 30, 40]]];
    it('3D', function () {
        var x = tensor3d(array3DData, [1, 1, 4]);
        expectTensorsClose(K.sliceAlongLastAxis(x, 1, 2), tensor3d([[[20, 30]]], [1, 1, 2]));
    });
    var array4DData = [[[[10, 20, 30, 40]]]];
    it('3D', function () {
        var x = tensor4d(array4DData, [1, 1, 1, 4]);
        expectTensorsClose(K.sliceAlongLastAxis(x, 1, 2), tensor4d([[[[20, 30]]]], [1, 1, 1, 2]));
    });
});
describeMathCPUAndGPU('sliceAlongAxis', function () {
    it('1D', function () {
        var array1DData = [10, 20, 30, 40];
        var x = tensor1d(array1DData);
        expectTensorsClose(K.sliceAlongAxis(x, 1, 2, 1), tensor1d([20, 30]));
    });
    var array2DData = [[10, 11], [20, 21], [30, 31], [40, 41]];
    it('2D-1', function () {
        var x = tensor2d(array2DData, [4, 2]);
        expectTensorsClose(K.sliceAlongAxis(x, 1, 2, 1), tensor2d([[20, 21], [30, 31]], [2, 2]));
    });
    it('2D-2', function () {
        var x = tensor2d(array2DData, [4, 2]);
        expectTensorsClose(K.sliceAlongAxis(x, 0, 1, 2), tensor2d([[10], [20], [30], [40]], [4, 1]));
    });
    var array3DData = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
    it('3D-1', function () {
        var x = tensor3d(array3DData, [2, 2, 2]);
        expectTensorsClose(K.sliceAlongAxis(x, 0, 1, 1), tensor3d([[[1, 2], [3, 4]]], [1, 2, 2]));
    });
    it('3D-2', function () {
        var x = tensor3d(array3DData, [2, 2, 2]);
        expectTensorsClose(K.sliceAlongAxis(x, 0, 1, 2), tensor3d([[[1, 2]], [[5, 6]]], [2, 1, 2]));
    });
    it('3D-3', function () {
        var x = tensor3d(array3DData, [2, 2, 2]);
        expectTensorsClose(K.sliceAlongAxis(x, 0, 1, 3), tensor3d([[[1], [3]], [[5], [7]]], [2, 2, 1]));
    });
    it('4D', function () {
        var array4DData = [[[[10, 1]]], [[[20, 2]]], [[[30, 3]]], [[[40, 4]]]];
        var x = tensor4d(array4DData, [4, 1, 1, 2]);
        expectTensorsClose(K.sliceAlongAxis(x, 0, 1, 4), tensor4d([[[[10]]], [[[20]]], [[[30]]], [[[40]]]], [4, 1, 1, 1]));
    });
});
describeMathCPUAndGPU('normalizeBatchInTraining', function () {
    it('2D, no broadcasting', function () {
        var x = tensor2d([[1, 2, 3, 4], [2, 4, 6, 8], [12, 11, 10, 9]], [3, 4]);
        var gamma = tensor1d([1, 1, 1, 1]);
        var beta = tensor1d([0, 0, 0, 0]);
        var reductionAxes = [0];
        var _a = K.normalizeBatchInTraining(x, gamma, beta, reductionAxes), normed = _a[0], mean = _a[1], variance = _a[2];
        expectTensorsClose(normed, tensor2d([
            [-0.805371, -0.9502233, -1.1624058, -1.3885813],
            [-0.6040282, -0.4319197, -0.11624074, 0.46286058],
            [1.4093992, 1.3821429, 1.2786462, 0.92572117]
        ], [3, 4]));
        expectTensorsClose(mean, tensor1d([5.0, 5.6666665, 6.3333335, 7.0]));
        expectTensorsClose(variance, tensor1d([24.666666, 14.888889, 8.222222, 4.6666665]));
    });
    it('3D, no broadcasting', function () {
        var x = tensor3d([[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
        var gamma = tensor1d([1, 1]);
        var beta = tensor1d([0, 0]);
        var reductionAxes = [0, 1];
        var _a = K.normalizeBatchInTraining(x, gamma, beta, reductionAxes), normed = _a[0], mean = _a[1], variance = _a[2];
        expectTensorsClose(normed, tensor3d([
            [[-1.1355163, -1.3552775], [-0.6488664, -0.7297648]],
            [[-0.8921913, -0.7297648], [0.08110833, 0.5212605]],
            [[1.5410578, 1.4595294], [1.0544081, 0.8340168]]
        ], [3, 2, 2]));
        expectTensorsClose(mean, tensor1d([5.6666665, 6.3333335]));
        expectTensorsClose(variance, tensor1d([16.88889, 10.222222]));
    });
    it('3D, broadcasting', function () {
        var x = tensor3d([[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]], [3, 2, 2]);
        var gamma = tensor2d([[1, 1], [1, 1]], [2, 2]);
        var beta = tensor2d([[0, 0], [0, 0]], [2, 2]);
        var reductionAxes = [0];
        var _a = K.normalizeBatchInTraining(x, gamma, beta, reductionAxes), normed = _a[0], mean = _a[1], variance = _a[2];
        expectTensorsClose(normed, tensor3d([
            [[-0.805371, -0.9502233], [-1.1624058, -1.3885813]],
            [[-0.6040282, -0.4319197], [-0.11624074, 0.46286058]],
            [[1.4093992, 1.3821429], [1.2786462, 0.92572117]]
        ], [3, 2, 2]));
        expectTensorsClose(mean, tensor2d([[5, 5.6666665], [6.3333335, 7]], [2, 2]));
        expectTensorsClose(variance, tensor2d([[24.666666, 14.888889], [8.222222, 4.6666665]], [2, 2]));
    });
    it('4D, broadcasting', function () {
        var x = tensor4d([[[[1, 2], [3, 4]], [[2, 4], [6, 8]], [[12, 11], [10, 9]]]], [1, 3, 2, 2]);
        var gamma = tensor2d([[1, 1], [1, 1]], [2, 2]);
        var beta = tensor2d([[0, 0], [0, 0]], [2, 2]);
        var reductionAxes = [0, 1];
        var _a = K.normalizeBatchInTraining(x, gamma, beta, reductionAxes), normed = _a[0], mean = _a[1], variance = _a[2];
        expectTensorsClose(normed, tensor4d([[
                [[-0.805371, -0.9502233], [-1.1624058, -1.3885813]],
                [[-0.6040282, -0.4319197], [-0.11624074, 0.46286058]],
                [[1.4093992, 1.3821429], [1.2786462, 0.92572117]]
            ]], [1, 3, 2, 2]));
        expectTensorsClose(mean, tensor2d([[5, 5.6666665], [6.3333335, 7]], [2, 2]));
        expectTensorsClose(variance, tensor2d([[24.666666, 14.888889], [8.222222, 4.6666665]], [2, 2]));
    });
});
describeMathCPUAndGPU('concatenate', function () {
    it('1D', function () {
        var x = tensor1d([1, 2, 3, 4]);
        var y = tensor1d([-1, -2, -3, -4]);
        var expected = tensor1d([1, 2, 3, 4, -1, -2, -3, -4]);
        expectTensorsClose(K.concatenate([x, y]), expected);
        expectTensorsClose(K.concatenate([x, y], -1), expected);
        expectTensorsClose(K.concatenate([x, y], 0), expected);
    });
    it('2D', function () {
        var x = tensor2d([1, 2, 3, 4], [2, 2]);
        var y = tensor2d([-1, -2, -3, -4], [2, 2]);
        var expected = tensor2d([1, 2, -1, -2, 3, 4, -3, -4], [2, 4]);
        expectTensorsClose(K.concatenate([x, y]), expected);
        expectTensorsClose(K.concatenate([x, y], -1), expected);
        expectTensorsClose(K.concatenate([x, y], 1), expected);
        expected = tensor2d([1, 2, 3, 4, -1, -2, -3, -4], [4, 2]);
        expectTensorsClose(K.concatenate([x, y], 0), expected);
    });
    it('3D', function () {
        var x = tensor3d([1, 2, 3, 4], [2, 2, 1]);
        var y = tensor3d([-1, -2, -3, -4], [2, 2, 1]);
        var expected = tensor3d([1, -1, 2, -2, 3, -3, 4, -4], [2, 2, 2]);
        expectTensorsClose(K.concatenate([x, y]), expected);
        expectTensorsClose(K.concatenate([x, y], -1), expected);
        expectTensorsClose(K.concatenate([x, y], 2), expected);
        expected = tensor3d([1, 2, -1, -2, 3, 4, -3, -4], [2, 4, 1]);
        expectTensorsClose(K.concatenate([x, y], 1), expected);
        expected = tensor3d([1, 2, 3, 4, -1, -2, -3, -4], [4, 2, 1]);
        expectTensorsClose(K.concatenate([x, y], 0), expected);
    });
    it('3D', function () {
        var x = tensor4d([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);
        var y = tensor4d([-1, -2, -3, -4, -5, -6, -7, -8], [2, 2, 2, 1]);
        var expected = tensor4d([1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8], [2, 2, 2, 2]);
        expectTensorsClose(K.concatenate([x, y]), expected);
        expectTensorsClose(K.concatenate([x, y], -1), expected);
        expectTensorsClose(K.concatenate([x, y], 3), expected);
        expected = tensor4d([1, 2, -1, -2, 3, 4, -3, -4, 5, 6, -5, -6, 7, 8, -7, -8], [2, 2, 4, 1]);
        expectTensorsClose(K.concatenate([x, y], 2), expected);
        expected = tensor4d([1, 2, 3, 4, -1, -2, -3, -4, 5, 6, 7, 8, -5, -6, -7, -8], [2, 4, 2, 1]);
        expectTensorsClose(K.concatenate([x, y], 1), expected);
        expected = tensor4d([1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8], [4, 2, 2, 1]);
        expectTensorsClose(K.concatenate([x, y], 0), expected);
    });
});
describeMathCPUAndGPU('concatAlongFirstAxis', function () {
    var array1DData1 = [10, 20, 30, 40];
    var array1DData2 = [-10, -20, -30, -40];
    it('1D', function () {
        var a = tensor1d(array1DData1);
        var b = tensor1d(array1DData2);
        expectTensorsClose(K.concatAlongFirstAxis(a, b), tensor1d([10, 20, 30, 40, -10, -20, -30, -40]));
    });
    var array2DData1 = [[10, 11], [20, 21]];
    var array2DData2 = [[30, 31], [40, 41]];
    it('2D', function () {
        var a = tensor2d(array2DData1, [2, 2]);
        var b = tensor2d(array2DData2, [2, 2]);
        expectTensorsClose(K.concatAlongFirstAxis(a, b), tensor2d([[10, 11], [20, 21], [30, 31], [40, 41]], [4, 2]));
    });
    var array3DData1 = [[[10]], [[20]]];
    var array3DData2 = [[[30]], [[40]]];
    it('3D', function () {
        var a = tensor3d(array3DData1, [2, 1, 1]);
        var b = tensor3d(array3DData2, [2, 1, 1]);
        expectTensorsClose(K.concatAlongFirstAxis(a, b), tensor3d([[[10]], [[20]], [[30]], [[40]]], [4, 1, 1]));
    });
    var array4DData1 = [[[[10]]], [[[20]]]];
    var array4DData2 = [[[[30]]], [[[40]]]];
    it('4D', function () {
        var a = tensor4d(array4DData1, [2, 1, 1, 1]);
        var b = tensor4d(array4DData2, [2, 1, 1, 1]);
        expectTensorsClose(K.concatAlongFirstAxis(a, b), tensor4d([[[[10]]], [[[20]]], [[[30]]], [[[40]]]], [4, 1, 1, 1]));
    });
    it('Scalar leads to error', function () {
        expect(function () {
            K.concatAlongFirstAxis(scalar(24), scalar(-24));
        }).toThrow();
    });
});
describeMathCPUAndGPU('tile', function () {
    it('1D, n is number', function () {
        var x = tensor1d([1, 3, 3, 7]);
        var n = 3;
        var y = K.tile(x, n);
        expectTensorsClose(y, tensor1d([1, 3, 3, 7, 1, 3, 3, 7, 1, 3, 3, 7]));
    });
    it('1D, n is number Array', function () {
        var x = tensor1d([1, 3, 3, 7]);
        var n = [3];
        var y = K.tile(x, n);
        expectTensorsClose(y, tensor1d([1, 3, 3, 7, 1, 3, 3, 7, 1, 3, 3, 7]));
    });
    it('2D', function () {
        var x = tensor2d([[1, 3], [3, 7]], [2, 2]);
        var n = [2, 3];
        var y = K.tile(x, n);
        expectTensorsClose(y, tensor2d([
            [1, 3, 1, 3, 1, 3], [3, 7, 3, 7, 3, 7], [1, 3, 1, 3, 1, 3],
            [3, 7, 3, 7, 3, 7]
        ], [4, 6]));
    });
    it('3D', function () {
        var x = tensor3d([[[1]]], [1, 1, 1]);
        var n = [2, 3, 4];
        var y = K.tile(x, n);
        expectTensorsClose(y, tfc.ones([2, 3, 4]));
    });
    it('Mismatch in x dimensions and n length leads to exception', function () {
        expect(function () { return K.tile(tfc.zeros([2, 2]), 1); })
            .toThrowError(/The length of input n \(1\) does not match .*2/);
    });
});
describeMathCPUAndGPU('Identity', function () {
    it('Scalar', function () {
        var s = scalar(12);
        var sIdentity = K.identity(s);
        expect(sIdentity.shape).toEqual([]);
        expect(sIdentity.dataSync()).toEqual(new Float32Array([12]));
    });
    it('1D', function () {
        var v = tensor1d([-12, 12]);
        var vIdentity = K.identity(v);
        expect(vIdentity.shape).toEqual([2]);
        expect(vIdentity.dataSync()).toEqual(new Float32Array([-12, 12]));
    });
    it('2D', function () {
        var m = tensor2d([[-12, 12], [-10, 10]], [2, 2]);
        var mIdentity = K.identity(m);
        expect(mIdentity.shape).toEqual([2, 2]);
        expect(mIdentity.dataSync()).toEqual(new Float32Array([-12, 12, -10, 10]));
    });
});
describeMathCPUAndGPU('scalarTimesArray', function () {
    it('Scalar x Scalar', function () {
        expectTensorsClose(K.scalarTimesArray(scalar(-2), scalar(-3)), scalar(6));
    });
    it('Scalar x 4D', function () {
        var y = K.scalarTimesArray(scalar(-2), tfc.ones([2, 2, 2, 2]));
        expect(y.shape).toEqual([2, 2, 2, 2]);
        var yValues = Array.from(y.dataSync());
        expect(unique(yValues)).toEqual([-2]);
    });
});
describeMathCPUAndGPU('scalarPlusArray', function () {
    it('Scalar + Scalar', function () {
        expectTensorsClose(K.scalarPlusArray(scalar(-2), scalar(-3)), scalar(-5));
    });
    it('Scalar + 4D', function () {
        var shape = [2, 2, 2, 2];
        var y = K.scalarPlusArray(scalar(-1), tfc.ones(shape));
        expectTensorsClose(y, tfc.zeros(shape));
    });
});
describeMathCPUAndGPU('randomNormal', function () {
    var dtypes = ['float32', 'int32'];
    var _loop_1 = function (dtype) {
        it("Scalar " + dtype, function () {
            var s = K.randomNormal([], 0, 10, dtype);
            expect(K.shape(s)).toEqual([]);
        });
        it("1D " + dtype, function () {
            var v = K.randomNormal([20], 0, 2, dtype);
            expect(K.shape(v)).toEqual([20]);
        });
        it("2D " + dtype, function () {
            var x = K.randomNormal([3, 20], -10, 20, dtype);
            expect(K.shape(x)).toEqual([3, 20]);
        });
        it("3D " + dtype, function () {
            var y = K.randomNormal([2, 3, 4], 100, 10, dtype);
            expect(K.shape(y)).toEqual([2, 3, 4]);
        });
    };
    for (var _i = 0, dtypes_1 = dtypes; _i < dtypes_1.length; _i++) {
        var dtype = dtypes_1[_i];
        _loop_1(dtype);
    }
});
describeMathCPUAndGPU('dot', function () {
    it('2D x 2D', function () {
        var x = tensor2d([[1, 0], [0, -1]], [2, 2]);
        var y = tensor2d([[3], [4]], [2, 1]);
        var output = K.dot(x, y);
        expectTensorsClose(output, tensor2d([[3], [-4]], [2, 1]));
    });
    it('2D x 2D: Incompatible dimensions', function () {
        var x = tensor2d([[1, 0], [0, -1]], [2, 2]);
        var y = tensor2d([[3], [4], [5]], [3, 1]);
        expect(function () { return K.dot(x, y); }).toThrowError();
    });
    it('3D x 2D', function () {
        var x = tensor3d([[[1, 0], [0, -1]], [[-2, 0], [0, -2]]], [2, 2, 2]);
        var y = tensor2d([[-1], [1]], [2, 1]);
        expectTensorsClose(K.dot(x, y), tensor3d([[[-1], [-1]], [[2], [-2]]], [2, 2, 1]));
    });
    it('2D x 1D leads to error', function () {
        var x = tensor2d([[1, 0], [0, -1]], [2, 2]);
        var y = tensor1d([3, 4]);
        expect(function () { return K.dot(x, y); }).toThrowError();
    });
    it('2D x Scalar leads to error', function () {
        var x = tensor2d([[1]], [1, 1]);
        var y = scalar(10);
        expect(function () { return K.dot(x, y); }).toThrowError();
    });
    it('1D x 1D leads to error', function () {
        var x = tensor1d([1, 2]);
        var y = tensor1d([3, 4]);
        expect(function () { return K.dot(x, y); }).toThrowError();
    });
});
describeMathCPUAndGPU('sign', function () {
    it('Scalar', function () {
        expectTensorsClose(K.sign(scalar(0)), scalar(0));
        expectTensorsClose(K.sign(scalar(0.5)), scalar(1));
        expectTensorsClose(K.sign(scalar(-0.5)), scalar(-1));
    });
    it('1D', function () {
        var x = tensor1d([1, 2, -1, 0, 3, -4]);
        expectTensorsClose(K.sign(x), tensor1d([1, 1, -1, 0, 1, -1]));
    });
    it('2D', function () {
        var x = tensor2d([[1, 2, -1], [0, 3, -4]], [2, 3]);
        expectTensorsClose(K.sign(x), tensor2d([[1, 1, -1], [0, 1, -1]], [2, 3]));
    });
});
describeMathCPUAndGPU('qr', function () {
    it('1x1', function () {
        var x = tensor2d([[10]], [1, 1]);
        var _a = K.qr(x), q = _a[0], r = _a[1];
        expectTensorsClose(q, tensor2d([[-1]], [1, 1]));
        expectTensorsClose(r, tensor2d([[-10]], [1, 1]));
    });
    it('2x2', function () {
        var x = tensor2d([[1, 3], [-2, -4]], [2, 2]);
        var _a = K.qr(x), q = _a[0], r = _a[1];
        expectTensorsClose(q, tensor2d([[-0.4472, -0.8944], [0.8944, -0.4472]], [2, 2]));
        expectTensorsClose(r, tensor2d([[-2.2361, -4.9193], [0, -0.8944]], [2, 2]));
    });
    it('3x3', function () {
        var x = tensor2d([[1, 3, 2], [-2, 0, 7], [8, -9, 4]], [3, 3]);
        var _a = K.qr(x), q = _a[0], r = _a[1];
        expectTensorsClose(q, tensor2d([
            [-0.1204, 0.8729, 0.4729], [0.2408, -0.4364, 0.8669],
            [-0.9631, -0.2182, 0.1576]
        ], [3, 3]));
        expectTensorsClose(r, tensor2d([[-8.3066, 8.3066, -2.4077], [0, 4.5826, -2.1822], [0, 0, 7.6447]], [3, 3]));
    });
    it('3x2', function () {
        var x = tensor2d([[1, 2], [3, -3], [-2, 1]], [3, 2]);
        var _a = K.qr(x), q = _a[0], r = _a[1];
        expectTensorsClose(q, tensor2d([
            [-0.2673, 0.9221, 0.2798], [-0.8018, -0.3738, 0.4663],
            [0.5345, -0.0997, 0.8393]
        ], [3, 3]));
        expectTensorsClose(r, tensor2d([[-3.7417, 2.4054], [0, 2.8661], [0, 0]], [3, 2]));
    });
    it('does not leak memory', function () {
        var x = tensor2d([[1, 3], [-2, -4]], [2, 2]);
        K.qr(x);
        var numTensors = memory().numTensors;
        K.qr(x);
        expect(memory().numTensors).toEqual(numTensors + 2);
    });
    it('Incorrect shape leads to error', function () {
        var x = tensor2d([[1, 2, 3], [-3, -2, 1]], [2, 3]);
        expect(function () { return K.qr(x); }).toThrowError(/requires.*shape/);
    });
});
describeMathCPUAndGPU('OneHot', function () {
    it('Unsupported indices', function () {
        var numClasses = 2;
        var indices = tensor2d([[-12, 12], [-10, 10]], [2, 2]);
        expect(function () {
            K.oneHot(indices, numClasses);
        }).toThrowError();
    });
    it('Unsupported numClasses', function () {
        var numClasses = 1;
        var indices = tensor1d([2, 2]);
        expect(function () {
            K.oneHot(indices, numClasses);
        }).toThrowError();
    });
    it('Supported use case', function () {
        var numClasses = 5;
        var indices = tensor1d([1, 3]);
        expectTensorsClose(K.oneHot(indices, numClasses), tensor2d([[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]], [2, 5]));
    });
});
describeMathCPUAndGPU('Gather', function () {
    it('1D, Array of numbers with repeats', function () {
        expectTensorsClose(K.gather(tensor1d([0, 10, 20, 30]), [2, 2, 3, 1]), tensor1d([20, 20, 30, 10]));
    });
    it('2D, Array of numbers', function () {
        expectTensorsClose(K.gather(tensor2d([[10, 20], [30, 40], [50, 60]], [3, 2]), [2, 0]), tensor2d([[50, 60], [10, 20]], [2, 2]));
    });
    it('2D, Tensor1D', function () {
        expectTensorsClose(K.gather(tensor2d([[10, 20], [30, 40], [50, 60]], [3, 2]), tensor1d([2, 1])), tensor2d([[50, 60], [30, 40]], [2, 2]));
    });
    it('3D, Tensor1D', function () {
        expectTensorsClose(K.gather(tensor3d([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], [2, 2, 2]), tensor1d([1, 0])), tensor3d([[[50, 60], [70, 80]], [[10, 20], [30, 40]]], [2, 2, 2]));
    });
    it('2D, Non-default axis', function () {
        expectTensorsClose(K.gather(tensor2d([[10, 20], [30, 40], [50, 60]], [3, 2]), [1], 1), tensor2d([[20], [40], [60]], [3, 1]));
    });
});
describeMathCPUAndGPU('Square', function () {
    it('Element-wise square', function () {
        expectTensorsClose(K.square(tensor2d([[1, -2], [-3, 4]], [2, 2])), tensor2d([1, 4, 9, 16], [2, 2]));
    });
});
describeMathCPUAndGPU('Pow', function () {
    it('Element-wise Pow: Positive Scalar', function () {
        expectTensorsClose(K.pow(tensor2d([[1, 1.5], [2, 2.5]], [2, 2]), scalar(2, 'int32')), tensor2d([[1, 2.25], [4, 6.25]], [2, 2]));
    });
    it('Element-wise Pow: Negative Scalar', function () {
        expectTensorsClose(K.pow(tensor2d([[1, 1.5], [2, 2.5]], [2, 2]), scalar(-2, 'int32')), tensor2d([[1, 1 / (1.5 * 1.5)], [1 / (2 * 2), 1 / (2.5 * 2.5)]], [2, 2]));
    });
    it('Element-wise Pow: Zero Scalar', function () {
        expectTensorsClose(K.pow(tensor2d([[1, 1.5], [2, 2.5]], [2, 2]), scalar(0, 'int32')), tensor2d([[1, 1], [1, 1]], [2, 2]));
    });
    it('Element-wise Pow: number', function () {
        expectTensorsClose(K.pow(tensor2d([[1, 1.5], [2, 2.5]], [2, 2]), 2), tensor2d([[1, 2.25], [4, 6.25]], [2, 2]));
    });
});
describeMathCPUAndGPU('softsign', function () {
    it('Element-wise softsign', function () {
        expectTensorsClose(tfc.tanh(tensor2d([[-2, -1], [1, 2]], [2, 2])), tensor2d([Math.tanh(-2), Math.tanh(-1), Math.tanh(1), Math.tanh(2)], [2, 2]));
    });
});
describeMathCPUAndGPU('batchNormalization', function () {
    it('2D, no broadcast, no gamma, no beta', function () {
        var x = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
        var variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, null, null, 0), tensor2d([[2.5, 3.75], [12.5, 8.75]], [2, 2]));
    });
    it('2D, no broadcast, no gamma, no beta, custom epsilon', function () {
        var x = tensor2d([[30, 30], [60, 60]], [2, 2]);
        var mean = tensor2d([[0, 0], [0, 0]], [2, 2]);
        var variance = tensor2d([[7, 7], [7, 7]], [2, 2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, null, null, 2), tensor2d([[10, 10], [20, 20]], [2, 2]));
    });
    it('2D, no broadcast, gamma, no beta', function () {
        var x = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
        var variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
        var gamma = tensor2d([[1, 2], [3, 4]], [2, 2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, null, gamma, 0), tensor2d([[2.5, 7.5], [37.5, 35]], [2, 2]));
    });
    it('2D, no broadcast, gamma, beta', function () {
        var x = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var mean = tensor2d([[5, 5], [5, 5]], [2, 2]);
        var variance = tensor2d([[4, 16], [4, 16]], [2, 2]);
        var gamma = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var beta = tensor2d([[-1, -1], [-2, -2]], [2, 2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, beta, gamma, 0), tensor2d([[1.5, 6.5], [35.5, 33]], [2, 2]));
    });
    it('2D, broadcast, gamma, beta', function () {
        var x = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var mean = tensor1d([2, 5]);
        var variance = tensor1d([1, 4]);
        var gamma = tensor1d([3, 4]);
        var beta = tensor1d([-1, -2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, beta, gamma, 0), tensor2d([[23, 28], [83, 68]], [2, 2]));
    });
    it('3D, no broadcast, no gamma, no beta', function () {
        var x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
        var mean = tensor3d([[[5, 5], [5, 5]], [[5, 5], [5, 5]]], [2, 2, 2]);
        var variance = tensor3d([[[4, 16], [4, 16]], [[16, 25], [16, 25]]], [2, 2, 2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, null, null, 0), tensor3d([[[2.5, 3.75], [12.5, 8.75]], [[1.25, 3], [6.25, 7]]], [2, 2, 2]));
    });
    it('3D, no broadcast, gamma, beta', function () {
        var x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
        var mean = tensor3d([[[5, 5], [5, 5]], [[5, 5], [5, 5]]], [2, 2, 2]);
        var variance = tensor3d([[[4, 16], [4, 16]], [[16, 25], [16, 25]]], [2, 2, 2]);
        var gamma = tensor3d([[[2, 2], [2, 2]], [[4, 4], [4, 4]]], [2, 2, 2]);
        var beta = tensor3d([[[-1, -1], [-2, -2]], [[-1, -1], [-2, -2]]], [2, 2, 2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, beta, gamma, 0), tensor3d([[[4, 6.5], [23, 15.5]], [[4, 11], [23, 26]]], [2, 2, 2]));
    });
    it('3D, broadcast, gamma, beta', function () {
        var x = tensor3d([[[10, 20], [30, 40]], [[10, 20], [30, 40]]], [2, 2, 2]);
        var mean = tensor1d([5, 5]);
        var variance = tensor1d([4, 16]);
        var gamma = tensor1d([2, 4]);
        var beta = tensor1d([-1, -2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, beta, gamma, 0), tensor3d([[[4, 13], [24, 33]], [[4, 13], [24, 33]]], [2, 2, 2]));
    });
    it('4D, no broadcast, no gamma, no beta', function () {
        var x = tensor4d([
            [[[10, 20], [30, 40]], [[10, 20], [30, 40]]],
            [[[-10, -20], [-30, -40]], [[-10, -20], [-30, -40]]]
        ], [2, 2, 2, 2]);
        var mean = tensor4d([
            [[[5, 5], [5, 5]], [[5, 5], [5, 5]]],
            [[[-5, -5], [-5, -5]], [[-5, -5], [-5, -5]]]
        ], [2, 2, 2, 2]);
        var variance = tensor4d([
            [[[4, 16], [4, 16]], [[16, 25], [16, 25]]],
            [[[4, 16], [4, 16]], [[16, 25], [16, 25]]]
        ], [2, 2, 2, 2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, null, null, 0), tensor4d([
            [[[2.5, 3.75], [12.5, 8.75]], [[1.25, 3], [6.25, 7]]],
            [[[-2.5, -3.75], [-12.5, -8.75]], [[-1.25, -3], [-6.25, -7]]]
        ], [2, 2, 2, 2]));
    });
    it('4D, no broadcast, gamma, beta', function () {
        var x = tensor4d([
            [[[10, 20], [30, 40]], [[10, 20], [30, 40]]],
            [[[-10, -20], [-30, -40]], [[-10, -20], [-30, -40]]]
        ], [2, 2, 2, 2]);
        var mean = tensor4d([
            [[[5, 5], [5, 5]], [[5, 5], [5, 5]]],
            [[[-5, -5], [-5, -5]], [[-5, -5], [-5, -5]]]
        ], [2, 2, 2, 2]);
        var variance = tensor4d([
            [[[4, 16], [4, 16]], [[16, 25], [16, 25]]],
            [[[4, 16], [4, 16]], [[16, 25], [16, 25]]]
        ], [2, 2, 2, 2]);
        var gamma = tensor4d([
            [[[2, 2], [2, 2]], [[4, 4], [4, 4]]],
            [[[2, 2], [2, 2]], [[4, 4], [4, 4]]]
        ], [2, 2, 2, 2]);
        var beta = tensor4d([
            [[[-1, -1], [-2, -2]], [[-1, -1], [-2, -2]]],
            [[[1, 1], [2, 2]], [[1, 1], [2, 2]]]
        ], [2, 2, 2, 2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, beta, gamma, 0), tensor4d([
            [[[4, 6.5], [23, 15.5]], [[4, 11], [23, 26]]],
            [[[-4, -6.5], [-23, -15.5]], [[-4, -11], [-23, -26]]]
        ], [2, 2, 2, 2]));
    });
    it('4D, broadcast, gamma, beta', function () {
        var x = tensor4d([[[[10, 20], [30, 40]]], [[[10, 20], [30, 40]]]], [2, 1, 2, 2]);
        var mean = tensor1d([5, 5]);
        var variance = tensor1d([4, 16]);
        var gamma = tensor1d([2, 4]);
        var beta = tensor1d([-1, -2]);
        expectTensorsClose(K.batchNormalization(x, mean, variance, beta, gamma, 0), tensor4d([[[[4, 13], [24, 33]]], [[[4, 13], [24, 33]]]], [2, 1, 2, 2]));
    });
});
describeMathCPUAndGPU('dropout', function () {
    var dropoutLevels = [0, 0.75];
    var _loop_2 = function (dropoutLevel) {
        it("Level = " + dropoutLevel, function () {
            var x = tensor2d(range(1, 21), [10, 2]);
            var y = K.dropout(x, scalar(dropoutLevel));
            expect(y.dtype).toEqual(x.dtype);
            expect(y.shape).toEqual(x.shape);
            var xValue = x.dataSync();
            var yValue = y.dataSync();
            var nKept = 0;
            for (var i = 0; i < xValue.length; ++i) {
                if (yValue[i] !== 0) {
                    nKept++;
                    expect(yValue[i]).toBeCloseTo(1 / (1 - dropoutLevel) * xValue[i]);
                }
            }
            var numel = K.countParams(x);
            if (dropoutLevel === 0) {
                expect(nKept).toEqual(numel);
            }
            else {
                expect(nKept).toBeLessThan(numel);
            }
        });
    };
    for (var _i = 0, dropoutLevels_1 = dropoutLevels; _i < dropoutLevels_1.length; _i++) {
        var dropoutLevel = dropoutLevels_1[_i];
        _loop_2(dropoutLevel);
    }
});
describeMathCPUAndGPU('l2Normalize', function () {
    it('normalizes with no axis defined.', function () {
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var norm = Math.sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4);
        var expected = tensor2d([[1 / norm, 2 / norm], [3 / norm, 4 / norm]], [2, 2]);
        var result = K.l2Normalize(x);
        expectTensorsClose(result, expected);
    });
    it('normalizes along axis = -1.', function () {
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var firstNorm = Math.sqrt(1 * 1 + 2 * 2);
        var secondNorm = Math.sqrt(3 * 3 + 4 * 4);
        var expected = tensor2d([[1 / firstNorm, 2 / firstNorm], [3 / secondNorm, 4 / secondNorm]], [2, 2]);
        var result = K.l2Normalize(x, -1);
        expectTensorsClose(result, expected);
    });
    it('normalizes with zeros.', function () {
        var x = zeros([2, 2]);
        var result = K.l2Normalize(x);
        expectTensorsClose(result, x);
    });
});
describeMathCPUAndGPU('biasAdd', function () {
    it('1D + 1D', function () {
        var x = tfc.ones([2]);
        var y = tensor1d([-1, 1]);
        expectTensorsClose(K.biasAdd(x, y), tensor1d([0, 2]));
    });
    it('2D + 1D', function () {
        var x = tfc.ones([2, 2]);
        var y = tensor1d([-1, 1]);
        expectTensorsClose(K.biasAdd(x, y), tensor2d([[0, 2], [0, 2]], [2, 2]));
    });
    it('3D + 1D', function () {
        var x = tfc.ones([2, 2, 2]);
        var y = tensor1d([-1, 1]);
        expectTensorsClose(K.biasAdd(x, y), tensor3d([[[0, 2], [0, 2]], [[0, 2], [0, 2]]], [2, 2, 2]));
    });
    it('4D + 1D', function () {
        var x = tfc.ones([1, 2, 2, 2]);
        var y = tensor1d([-1, 1]);
        expectTensorsClose(K.biasAdd(x, y), tensor4d([[[[0, 2], [0, 2]], [[0, 2], [0, 2]]]], [1, 2, 2, 2]));
    });
    it('2D + 1D: Incompatible size', function () {
        var x = tfc.ones([2, 2]);
        var y = tensor1d([-1, 0, 1]);
        expect(function () { return K.biasAdd(x, y); }).toThrowError();
    });
    it('3D + 2D leads to error', function () {
        var x = tfc.ones([2, 2, 2]);
        var y = tfc.ones([2, 2]);
        expect(function () { return K.biasAdd(x, y); }).toThrowError();
    });
});
describeMathCPUAndGPU('elu', function () {
    it('elu', function () {
        var xData = [-1, 0, 1, -1];
        expectTensorsClose(K.elu(tensor2d(xData, [2, 2])), tensor2d(xData.map(function (x) { return x < 0 ? Math.exp(x) - 1 : x; }), [2, 2]));
    });
});
describeMathCPUAndGPU('softsign', function () {
    it('softsign', function () {
        var xData = [-1, 0, 1, -1];
        expectTensorsClose(K.softsign(tensor2d(xData, [2, 2])), tensor2d(xData.map(function (x) { return x / (Math.abs(x) + 1); }), [2, 2]));
    });
    it('Does not leak', function () {
        var input = tensor2d([-1, 0, 1, -1], [2, 2]);
        expectNoLeakedTensors(function () { return K.softsign(input); }, 1);
    });
});
describe('floatx ', function () {
    it('returns "float32"', function () {
        expect(K.floatx()).toEqual('float32');
    });
});
describe('Name scope ', function () {
    it('returns function\'s value from the name scope.', function () {
        var name = 'name';
        var val = 'val';
        var fn = function () { return val; };
        expect(K.nameScope(name, fn)).toEqual(val);
    });
    it('re-throws exception.', function () {
        var exceptionValue = 'exception';
        var exceptionFn = function () {
            throw new Error(exceptionValue);
        };
        var nameScopeFn = function () {
            K.nameScope('foo', exceptionFn);
        };
        expect(nameScopeFn).toThrowError(exceptionValue);
    });
});
describe('getUID ', function () {
    it('second UID is different.', function () {
        var name = 'def';
        var firstUID = K.getUid(name);
        var secondUID = K.getUid(name);
        expect(secondUID).not.toEqual(firstUID);
    });
    it('with no prefix works and returns different UIDs.', function () {
        var firstUID = K.getUid();
        var secondUID = K.getUid();
        expect(firstUID).not.toEqual(secondUID);
    });
});
describeMathCPUAndGPU('categoricalCrossentropy ', function () {
    it('from logits', function () {
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
        var expected = tensor1d([
            -1 *
                (Math.log(Math.exp(1) / (Math.exp(1) + Math.exp(2))) * 0.25 +
                    Math.log(Math.exp(2) / (Math.exp(1) + Math.exp(2))) * 0.75),
            -1 *
                (Math.log(Math.exp(3) / (Math.exp(3) + Math.exp(4))) * 0.1 +
                    Math.log(Math.exp(4) / (Math.exp(3) + Math.exp(4))) * 0.9)
        ]);
        var result = K.categoricalCrossentropy(target, x, true);
        expectTensorsClose(result, expected);
    });
    it('from softmax', function () {
        var x = tensor2d([[0.3, 0.7], [0.4, 0.6]], [2, 2]);
        var target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
        var expected = tensor1d([
            -1 * (Math.log(0.3) * 0.25 + Math.log(0.7) * 0.75),
            -1 * (Math.log(0.4) * 0.1 + Math.log(0.6) * 0.9)
        ]);
        var result = K.categoricalCrossentropy(target, x, false);
        expectTensorsClose(result, expected);
    });
});
describeMathCPUAndGPU('sparseCategoricalCrossentropy ', function () {
    it('from logits', function () {
        var x = tensor2d([[1, 2, 3], [4, 5, 6]], [2, 3]);
        var target = tensor1d([0, 2]);
        var expected = tensor1d([
            -1 * Math.log(Math.exp(1) / (Math.exp(1) + Math.exp(2) + Math.exp(3))),
            -1 * Math.log(Math.exp(6) / (Math.exp(4) + Math.exp(5) + Math.exp(6)))
        ]);
        var result = K.sparseCategoricalCrossentropy(target, x, true);
        expectTensorsClose(result, expected);
    });
    it('from softmax', function () {
        var x = tensor2d([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]], [2, 3]);
        var target = tensor1d([0, 2]);
        var expected = tensor1d([-1 * Math.log(0.1), -1 * Math.log(0.5)]);
        var result = K.sparseCategoricalCrossentropy(target, x, false);
        expectTensorsClose(result, expected);
    });
});
describeMathCPUAndGPU('binaryCrossentropy', function () {
    function _binaryCrossentropy(target, output) {
        var targetComplement = K.scalarPlusArray(scalar(1), tfc.neg(target));
        var outputComplement = K.scalarPlusArray(scalar(1), tfc.neg(output));
        return tfc.neg(tfc.add(tfc.mul(target, tfc.log(output)), tfc.mul(targetComplement, tfc.log(outputComplement))));
    }
    it('from logits', function () {
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
        var sigmoidX = tfc.sigmoid(x);
        var expected = _binaryCrossentropy(target, sigmoidX);
        var result = K.binaryCrossentropy(target, x, true);
        expectTensorsClose(result, expected);
    });
    it('from sigmoid', function () {
        var x = tensor2d([[0.3, 0.7], [0.4, 0.6]], [2, 2]);
        var target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
        var expected = _binaryCrossentropy(target, x);
        var result = K.binaryCrossentropy(target, x, false);
        expectTensorsClose(result, expected);
    });
});
describeMathCPUAndGPU('sigmoidCrossEntropyWithLogits', function () {
    it('outputs sigmoid cross-entropy', function () {
        var x = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var target = tensor2d([[0.25, 0.75], [0.1, 0.9]], [2, 2]);
        var targetComplement = K.scalarPlusArray(scalar(1), tfc.neg(target));
        var sigmoidX = tfc.sigmoid(x);
        var sigmoidXComplement = K.scalarPlusArray(scalar(1), tfc.neg(sigmoidX));
        var expected = tfc.add(tfc.mul(target, tfc.neg(tfc.log(sigmoidX))), tfc.mul(targetComplement, tfc.neg(tfc.log(sigmoidXComplement))));
        var result = K.sigmoidCrossEntropyWithLogits(target, x);
        expectTensorsClose(result, expected);
    });
});
describeMathCPUAndGPU('Sigmoid', function () {
    it('2D', function () {
        var xValues = [-5, -2, 0, 1, 2, 5];
        var x = tensor2d(xValues, [2, 3]);
        var y = tfc.sigmoid(x);
        var yValuesExpected = xValues.map(function (v) { return 1 / (1 + Math.exp(-v)); });
        expectTensorsClose(y, tensor2d(yValuesExpected, [2, 3]));
    });
});
describeMathCPUAndGPU('hardSigmoid', function () {
    it('2D', function () {
        var xValues = [-5, -2, 0, 1, 2, 5];
        var x = tensor2d(xValues, [2, 3]);
        var y = K.hardSigmoid(x);
        var yValuesExpected = xValues.map(function (x) {
            var y = 0.2 * x + 0.5;
            if (y > 1) {
                return 1;
            }
            else if (y < 0) {
                return 0;
            }
            else {
                return y;
            }
        });
        expectTensorsClose(y, tensor2d(yValuesExpected, [2, 3]));
    });
});
describe('inTrainPhase', function () {
    it('training = true', function () {
        expect(K.inTrainPhase(function () { return -5; }, function () { return 5; }, true)).toEqual(-5);
    });
    it('training = false', function () {
        expect(K.inTrainPhase(function () { return -5; }, function () { return 5; }, false)).toEqual(5);
    });
    it('training = default false', function () {
        expect(K.inTrainPhase(function () { return -5; }, function () { return 5; })).toEqual(5);
    });
});
describeMathCPUAndGPU('gradients', function () {
    it('Simple mean: 1 variable', function () {
        var var1 = new LayerVariable(K.scalarTimesArray(scalar(2.0), tfc.ones([2, 2])));
        var gradients = K.gradients(function () { return tfc.mean(var1.read()); }, [var1]);
        expect(gradients.length).toEqual(1);
        expectTensorsClose(tensor2d([[0.25, 0.25], [0.25, 0.25]], [2, 2]), gradients[0]);
    });
    it('Simple matmul and mean: 2 variables', function () {
        var var1 = new LayerVariable(tensor2d([[1, 0], [0, 0]], [2, 2]));
        var var2 = new LayerVariable(tensor2d([[1, 0], [0, 1]], [2, 2]));
        var gradients = K.gradients(function () { return tfc.mean(K.dot(var1.read(), var2.read())); }, [var1, var2]);
        expect(gradients.length).toEqual(2);
        expectTensorsClose(tensor2d([[0.25, 0.25], [0.25, 0.25]], [2, 2]), gradients[0]);
        expectTensorsClose(tensor2d([[0.25, 0.25], [0, 0]], [2, 2]), gradients[1]);
    });
});
//# sourceMappingURL=tfjs_backend_test.js.map
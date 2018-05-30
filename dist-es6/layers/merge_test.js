import { tensor2d, tensor3d } from '@tensorflow/tfjs-core';
import * as tfl from '../index';
import { deserialize } from '../layers/serialization';
import { convertPythonicToTs } from '../utils/serialization_utils';
import { describeMathCPU, describeMathCPUAndGPU, expectTensorsClose } from '../utils/test_utils';
import { Add, Average, Concatenate, Maximum, Minimum, Multiply } from './merge';
describeMathCPU('Merge Layers Except Concatenate: Symbolic', function () {
    var layers = [Add, Average, Multiply, Maximum, Minimum];
    var symbolicInputShapes = [
        [10, 3],
        [10, 2, 2],
    ];
    var numInputsArray = [2, 4];
    var _loop_1 = function (layer) {
        var _loop_2 = function (inputShape) {
            var _loop_3 = function (numInputs) {
                var testTitle = "layer=" + layer.name + "; inputShape=" + JSON.stringify(inputShape) + "; " +
                    ("numInputs=" + numInputs);
                it(testTitle, function () {
                    var addLayer = new layer({ name: layer.name });
                    var symbolicInputs = [];
                    for (var i = 0; i < numInputs; ++i) {
                        symbolicInputs.push(new tfl.SymbolicTensor('float32', inputShape, null, [], null));
                    }
                    var output = addLayer.apply(symbolicInputs);
                    expect(output.dtype).toEqual(symbolicInputs[0].dtype);
                    expect(output.shape).toEqual(inputShape);
                });
            };
            for (var _i = 0, numInputsArray_1 = numInputsArray; _i < numInputsArray_1.length; _i++) {
                var numInputs = numInputsArray_1[_i];
                _loop_3(numInputs);
            }
        };
        for (var _i = 0, symbolicInputShapes_1 = symbolicInputShapes; _i < symbolicInputShapes_1.length; _i++) {
            var inputShape = symbolicInputShapes_1[_i];
            _loop_2(inputShape);
        }
    };
    for (var _i = 0, layers_1 = layers; _i < layers_1.length; _i++) {
        var layer = layers_1[_i];
        _loop_1(layer);
    }
    it('Single input leads to exception', function () {
        var x = new tfl.SymbolicTensor('float32', [2, 2], null, [], null);
        var addLayer = new Add({ name: 'Add' });
        expect(function () {
            addLayer.apply([x]);
        }).toThrowError(/.*at least 2 inputs\. Got 1 input.*/);
    });
    it('Non-unique batch sizes to exception', function () {
        var x1 = new tfl.SymbolicTensor('float32', [1, 2], null, [], null);
        var x2 = new tfl.SymbolicTensor('float32', [2, 2], null, [], null);
        var addLayer = new Add({ name: 'Add' });
        expect(function () {
            addLayer.apply([x1, x2]);
        }).toThrowError(/Can not merge tensors with different batch sizes/);
    });
});
describeMathCPUAndGPU('Add-Functional', function () {
    it('Calling without arg returns Layer', function () {
        expect(tfl.layers.add().getClassName()).toEqual('Add');
    });
    it('Calling with config arg returns Layer', function () {
        expect(tfl.layers.add({ name: 'addLayer' }).name.indexOf('addLayer'))
            .toEqual(0);
    });
    it('Calling with symbolic tensors returns symbolic tensor', function () {
        var input1 = tfl.layers.input({ shape: [2, 2] });
        var input2 = tfl.layers.input({ shape: [2, 2] });
        var output = tfl.layers.add().apply([input1, input2]);
        expect(output.shape).toEqual([null, 2, 2]);
    });
    it('Calling with tensors returns tensor', function () {
        var input1 = tensor2d([1, 2, 3, 4], [2, 2]);
        var input2 = tensor2d([10, 20, 30, 40], [2, 2]);
        var output = tfl.layers.add().apply([input1, input2]);
        expectTensorsClose(output, tensor2d([11, 22, 33, 44], [2, 2]));
    });
});
describeMathCPUAndGPU('Multiply-Functional', function () {
    it('Calling without arg returns Layer', function () {
        expect(tfl.layers.multiply().getClassName()).toEqual('Multiply');
    });
    it('Calling with config arg returns Layer', function () {
        expect(tfl.layers.multiply({ name: 'multiplyLayer' })
            .name.indexOf('multiplyLayer'))
            .toEqual(0);
    });
    it('Calling with symbolic tensors returns symbolic tensor', function () {
        var input1 = tfl.layers.input({ shape: [2, 2] });
        var input2 = tfl.layers.input({ shape: [2, 2] });
        var output = tfl.layers.multiply().apply([input1, input2]);
        expect(output.shape).toEqual([null, 2, 2]);
    });
    it('Calling with tensors returns tensor', function () {
        var input1 = tensor2d([1, 2, 3, 4], [2, 2]);
        var input2 = tensor2d([10, 20, 30, 40], [2, 2]);
        var output = tfl.layers.multiply().apply([input1, input2]);
        expectTensorsClose(output, tensor2d([10, 40, 90, 160], [2, 2]));
    });
});
describeMathCPUAndGPU('Average-Functional', function () {
    it('Calling without arg returns Layer', function () {
        expect(tfl.layers.average().getClassName()).toEqual('Average');
    });
    it('Calling with config arg returns Layer', function () {
        expect(tfl.layers.average({ name: 'averageLayer' })
            .name.indexOf('averageLayer'))
            .toEqual(0);
    });
    it('Calling with symbolic tensors returns symbolic tensor', function () {
        var input1 = tfl.layers.input({ shape: [2, 2] });
        var input2 = tfl.layers.input({ shape: [2, 2] });
        var output = tfl.layers.average().apply([input1, input2]);
        expect(output.shape).toEqual([null, 2, 2]);
    });
    it('Calling with tensors returns tensor', function () {
        var input1 = tensor2d([1, 2, 3, 4], [2, 2]);
        var input2 = tensor2d([10, 20, 30, 40], [2, 2]);
        var output = tfl.layers.average().apply([input1, input2]);
        expectTensorsClose(output, tensor2d([5.5, 11, 16.5, 22], [2, 2]));
    });
});
describeMathCPUAndGPU('Maximum-Functional', function () {
    it('Calling without arg returns Layer', function () {
        expect(tfl.layers.maximum().getClassName()).toEqual('Maximum');
    });
    it('Calling with config arg returns Layer', function () {
        expect(tfl.layers.maximum({ name: 'maximumLayer' })
            .name.indexOf('maximumLayer'))
            .toEqual(0);
    });
    it('Calling with symbolic tensors returns symbolic tensor', function () {
        var input1 = tfl.layers.input({ shape: [2, 2] });
        var input2 = tfl.layers.input({ shape: [2, 2] });
        var output = tfl.layers.maximum().apply([input1, input2]);
        expect(output.shape).toEqual([null, 2, 2]);
    });
    it('Calling with tensors returns tensor', function () {
        var input1 = tensor2d([1, 20, 3, 40], [2, 2]);
        var input2 = tensor2d([10, 2, 30, 4], [2, 2]);
        var output = tfl.layers.maximum().apply([input1, input2]);
        expectTensorsClose(output, tensor2d([10, 20, 30, 40], [2, 2]));
    });
});
describeMathCPUAndGPU('Minimum-Functional', function () {
    it('Calling without arg returns Layer', function () {
        expect(tfl.layers.minimum().getClassName()).toEqual('Minimum');
    });
    it('Calling with config arg returns Layer', function () {
        expect(tfl.layers.minimum({ name: 'minimumLayer' })
            .name.indexOf('minimumLayer'))
            .toEqual(0);
    });
    it('Calling with symbolic tensors returns symbolic tensor', function () {
        var input1 = tfl.layers.input({ shape: [2, 2] });
        var input2 = tfl.layers.input({ shape: [2, 2] });
        var output = tfl.layers.minimum().apply([input1, input2]);
        expect(output.shape).toEqual([null, 2, 2]);
    });
    it('Calling with tensors returns tensor', function () {
        var input1 = tensor2d([1, 20, 3, 40], [2, 2]);
        var input2 = tensor2d([10, 2, 30, 4], [2, 2]);
        var output = tfl.layers.minimum().apply([input1, input2]);
        expectTensorsClose(output, tensor2d([1, 2, 3, 4], [2, 2]));
    });
});
describeMathCPUAndGPU('Concatenate-Functional', function () {
    it('Calling without arg returns Layer', function () {
        expect(tfl.layers.concatenate().getClassName())
            .toEqual('Concatenate');
    });
    it('Calling with config arg returns Layer', function () {
        expect(tfl.layers.concatenate({ name: 'concatenateLayer' })
            .name.indexOf('concatenateLayer'))
            .toEqual(0);
    });
    it('Calling with symbolic tensors returns symbolic tensor', function () {
        var input1 = tfl.layers.input({ shape: [2, 3] });
        var input2 = tfl.layers.input({ shape: [2, 4] });
        var output = tfl.layers.concatenate().apply([input1, input2]);
        expect(output.shape).toEqual([null, 2, 7]);
    });
    it('Calling with tensors returns tensor', function () {
        var input1 = tensor2d([[1, 2], [3, 4]], [2, 2]);
        var input2 = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var output = tfl.layers.concatenate().apply([input1, input2]);
        expectTensorsClose(output, tensor2d([[1, 2, 10, 20], [3, 4, 30, 40]], [2, 4]));
    });
});
describeMathCPU('Concatenate Layer: Symbolic', function () {
    it('All known shapes', function () {
        var x1 = new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
        var x2 = new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
        var layer0 = new Concatenate({});
        expect(layer0.apply([x1, x2]).shape).toEqual([
            2, 3, 8
        ]);
        var layer1 = new Concatenate({ axis: -1 });
        expect(layer1.apply([x1, x2]).shape).toEqual([
            2, 3, 8
        ]);
        var layer2 = new Concatenate({ axis: 0 });
        expect(layer2.apply([x1, x2]).shape).toEqual([
            4, 3, 4
        ]);
        var layer3 = new Concatenate({ axis: 1 });
        expect(layer3.apply([x1, x2]).shape).toEqual([
            2, 6, 4
        ]);
    });
    it('Concat axis has unknown shape', function () {
        var x1 = new tfl.SymbolicTensor('float32', [2, null, 4], null, [], null);
        var x2 = new tfl.SymbolicTensor('float32', [2, null, 4], null, [], null);
        var layer = new Concatenate({ axis: 1 });
        expect(layer.apply([x1, x2]).shape).toEqual([
            2, null, 4
        ]);
    });
    it('Non-concat axis has unknown shape', function () {
        var x1 = new tfl.SymbolicTensor('float32', [null, 3, 4], null, [], null);
        var x2 = new tfl.SymbolicTensor('float32', [null, 5, 4], null, [], null);
        var layer = new Concatenate({ axis: 1 });
        expect(layer.apply([x1, x2]).shape).toEqual([
            null, 8, 4
        ]);
    });
    it('Incompatible shape leads to error', function () {
        var x1 = new tfl.SymbolicTensor('float32', [2, 3, 5], null, [], null);
        var x2 = new tfl.SymbolicTensor('float32', [2, 4, 5], null, [], null);
        var layer = new Concatenate({});
        expect(function () { return layer.apply([
            x1, x2
        ]); }).toThrowError(/requires inputs with matching shapes except/);
    });
    it('Single shape leads to error', function () {
        var x1 = new tfl.SymbolicTensor('float32', [2, 3, 5], null, [], null);
        var layer = new Concatenate({});
        expect(function () { return layer.apply([x1]); })
            .toThrowError(/should be called on a list of at least 2 inputs/);
    });
});
describeMathCPUAndGPU('Add Layer: Tensor', function () {
    it('2D plus 2D', function () {
        var x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var x2 = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
        var addLayer = new Add({});
        var y = addLayer.apply([x1, x2]);
        expectTensorsClose(y, tensor2d([[9, 18], [27, 36]], [2, 2]));
    });
    it('2D plus 2D, with broadcast', function () {
        var x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var x2 = tensor2d([[-2], [-4]], [2, 1]);
        var addLayer = new Add({});
        var y = addLayer.apply([x1, x2]);
        expectTensorsClose(y, tensor2d([[8, 18], [26, 36]], [2, 2]));
    });
    it('2D plus 2D, with dimension expansion', function () {
        var x1 = tensor3d([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], [2, 2, 2]);
        var x2 = tensor2d([[-2], [-4]], [2, 1]);
        var addLayer = new Add({});
        var y = addLayer.apply([x1, x2]);
        expectTensorsClose(y, tensor3d([[[8, 18], [28, 38]], [[46, 56], [66, 76]]], [2, 2, 2]));
    });
});
describeMathCPUAndGPU('Multiply Layer: Tensor', function () {
    it('2D times 2D', function () {
        var x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var x2 = tensor2d([[-1, -2], [-3, -4]], [2, 2]);
        var multipyLayer = new Multiply({});
        var y = multipyLayer.apply([x1, x2]);
        expectTensorsClose(y, tensor2d([[-10, -40], [-90, -160]], [2, 2]));
    });
});
describeMathCPUAndGPU('Average Layer: Tensor', function () {
    it('2D and 2D', function () {
        var x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var x2 = tensor2d([[-2, -4], [-6, -8]], [2, 2]);
        var averageLayer = new Average({});
        var y = averageLayer.apply([x1, x2]);
        expectTensorsClose(y, tensor2d([[4, 8], [12, 16]], [2, 2]));
    });
    it('2D and 2D, with broadcast', function () {
        var x1 = tensor2d([[10, 20], [30, 40]], [2, 2]);
        var x2 = tensor2d([[-2], [-4]], [2, 1]);
        var averageLayer = new Average({});
        var y = averageLayer.apply([x1, x2]);
        expectTensorsClose(y, tensor2d([[4, 9], [13, 18]], [2, 2]));
    });
});
describeMathCPUAndGPU('Maximum Layer: Tensor', function () {
    it('2D and 2D', function () {
        var x1 = tensor2d([[10, 20], [-6, -8]], [2, 2]);
        var x2 = tensor2d([[-2, -4], [30, 40]], [2, 2]);
        var averageLayer = new Maximum({});
        var y = averageLayer.apply([x1, x2]);
        expectTensorsClose(y, tensor2d([[10, 20], [30, 40]], [2, 2]));
    });
});
describeMathCPUAndGPU('Minimum Layer: Tensor', function () {
    it('2D and 2D', function () {
        var x1 = tensor2d([[10, 20], [-6, -8]], [2, 2]);
        var x2 = tensor2d([[-2, -4], [30, 40]], [2, 2]);
        var averageLayer = new Minimum({});
        var y = averageLayer.apply([x1, x2]);
        expectTensorsClose(y, tensor2d([[-2, -4], [-6, -8]], [2, 2]));
    });
});
describeMathCPUAndGPU('Concatenate Layer: Tensor', function () {
    var x1;
    var x2;
    function createData() {
        x1 = tensor2d([1, 2, 3, 4], [2, 2]);
        x2 = tensor2d([-1, -2, -3, -4], [2, 2]);
    }
    var axisValues = [null, undefined, 0, 1, -1];
    var _loop_4 = function (axis) {
        it("axis=" + axis, function () {
            createData();
            var layer = new Concatenate({ axis: axis });
            var expected = axis === 0 ?
                tensor2d([1, 2, 3, 4, -1, -2, -3, -4], [4, 2]) :
                tensor2d([1, 2, -1, -2, 3, 4, -3, -4], [2, 4]);
            expectTensorsClose(layer.apply([x1, x2]), expected);
        });
    };
    for (var _i = 0, axisValues_1 = axisValues; _i < axisValues_1.length; _i++) {
        var axis = axisValues_1[_i];
        _loop_4(axis);
    }
});
describeMathCPU('Deserialize Merge Layers', function () {
    it('Model with Add Layer', function () {
        var modelWithMergeJSON = {
            'class_name': 'Model',
            'keras_version': '2.1.5',
            'config': {
                'layers': [
                    {
                        'class_name': 'InputLayer',
                        'config': {
                            'dtype': 'float32',
                            'batch_input_shape': [null, 4],
                            'name': 'input_1',
                            'sparse': false
                        },
                        'inbound_nodes': [],
                        'name': 'input_1'
                    },
                    {
                        'class_name': 'InputLayer',
                        'config': {
                            'dtype': 'float32',
                            'batch_input_shape': [null, 4],
                            'name': 'input_2',
                            'sparse': false
                        },
                        'inbound_nodes': [],
                        'name': 'input_2'
                    },
                    {
                        'class_name': 'Add',
                        'config': { 'trainable': true, 'name': 'add_1' },
                        'inbound_nodes': [[['input_1', 0, 0, {}], ['input_2', 0, 0, {}]]],
                        'name': 'add_1'
                    }
                ],
                'input_layers': [['input_1', 0, 0], ['input_2', 0, 0]],
                'output_layers': [['add_1', 0, 0]],
                'name': 'model_1'
            },
            'backend': 'tensorflow'
        };
        var tsConfig = convertPythonicToTs(modelWithMergeJSON);
        var model = deserialize(tsConfig);
        expect(model.inputs.length).toEqual(2);
        expect(model.inputs[0].shape).toEqual([null, 4]);
        expect(model.inputs[1].shape).toEqual([null, 4]);
        expect(model.layers.length).toEqual(3);
        expect(model.layers[2] instanceof Add);
        expect(model.outputs.length).toEqual(1);
        expect(model.outputs[0].shape).toEqual([null, 4]);
    });
    it('Model with Concatenate Layer', function () {
        var modelWithMergeJSON = {
            'class_name': 'Model',
            'keras_version': '2.1.5',
            'config': {
                'layers': [
                    {
                        'class_name': 'InputLayer',
                        'config': {
                            'dtype': 'float32',
                            'batch_input_shape': [null, 4],
                            'name': 'input_1',
                            'sparse': false
                        },
                        'inbound_nodes': [],
                        'name': 'input_1'
                    },
                    {
                        'class_name': 'InputLayer',
                        'config': {
                            'dtype': 'float32',
                            'batch_input_shape': [null, 4],
                            'name': 'input_2',
                            'sparse': false
                        },
                        'inbound_nodes': [],
                        'name': 'input_2'
                    },
                    {
                        'class_name': 'Concatenate',
                        'config': { 'trainable': true, 'name': 'concatenate_1', 'axis': -1 },
                        'inbound_nodes': [[['input_1', 0, 0, {}], ['input_2', 0, 0, {}]]],
                        'name': 'concatenate_1'
                    }
                ],
                'input_layers': [['input_1', 0, 0], ['input_2', 0, 0]],
                'output_layers': [['concatenate_1', 0, 0]],
                'name': 'model_1'
            },
            'backend': 'tensorflow'
        };
        var tsConfig = convertPythonicToTs(modelWithMergeJSON);
        var model = deserialize(tsConfig);
        expect(model.inputs.length).toEqual(2);
        expect(model.inputs[0].shape).toEqual([null, 4]);
        expect(model.inputs[1].shape).toEqual([null, 4]);
        expect(model.layers.length).toEqual(3);
        expect(model.layers[2] instanceof Concatenate);
        expect(model.outputs.length).toEqual(1);
        expect(model.outputs[0].shape).toEqual([null, 8]);
    });
});
//# sourceMappingURL=merge_test.js.map
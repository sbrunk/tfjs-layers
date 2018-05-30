import { convertPythonicToTs, convertTsToPythonic } from './serialization_utils';
describe('convertPythonToTs', function () {
    it('primitives', function () {
        expect(convertPythonicToTs(null)).toEqual(null);
        expect(convertPythonicToTs(true)).toEqual(true);
        expect(convertPythonicToTs(false)).toEqual(false);
        expect(convertPythonicToTs(4)).toEqual(4);
    });
    it('strings w/o name tags', function () {
        expect(convertPythonicToTs('abc')).toEqual('abc');
        expect(convertPythonicToTs('ABC')).toEqual('ABC');
        expect(convertPythonicToTs('one_two')).toEqual('oneTwo');
        expect(convertPythonicToTs('OneTwo')).toEqual('OneTwo');
    });
    it('simple arrays', function () {
        expect(convertPythonicToTs([])).toEqual([]);
        expect(convertPythonicToTs([null])).toEqual([null]);
        expect(convertPythonicToTs(['one_two'])).toEqual(['oneTwo']);
        expect(convertPythonicToTs([null, true, false, 4, 'abc'])).toEqual([
            null, true, false, 4, 'abc'
        ]);
        expect(convertPythonicToTs([[[]]])).toEqual([[[]]]);
    });
    it('layer tuple (array) with key', function () {
        for (var _i = 0, _a = ['inboundNodes', 'inputLayers', 'outputLayers']; _i < _a.length; _i++) {
            var key = _a[_i];
            expect(convertPythonicToTs(['layer_name', 'meta_data', 0], key)).toEqual([
                'layer_name', 'metaData', 0
            ]);
        }
    });
    it('dictionary', function () {
        expect(convertPythonicToTs({})).toEqual({});
        expect(convertPythonicToTs({ key: null })).toEqual({ key: null });
        expect(convertPythonicToTs({ key_one: 4 })).toEqual({ keyOne: 4 });
        expect(convertPythonicToTs({ key_two: 'abc_def' })).toEqual({
            keyTwo: 'abcDef'
        });
        expect(convertPythonicToTs({ key_one: true, key_two: false }))
            .toEqual({ keyOne: true, keyTwo: false });
        expect(convertPythonicToTs({ name: 'layer_name' })).toEqual({
            name: 'layer_name'
        });
    });
    it('dictionary keys are passed down the stack', function () {
        var dict = { inbound_nodes: ['DoNotChange_Me', 0, null] };
        expect(convertPythonicToTs(dict)).toEqual({
            inboundNodes: ['DoNotChange_Me', 0, null]
        });
    });
    it('enum promotion', function () {
        expect(convertPythonicToTs({ mode: 'fan_out' })).toEqual({ mode: 'fanOut' });
        expect(convertPythonicToTs({ distribution: 'normal' })).toEqual({
            distribution: 'normal'
        });
        expect(convertPythonicToTs({ data_format: 'channels_last' })).toEqual({
            dataFormat: 'channelsLast'
        });
        expect(convertPythonicToTs({ padding: 'valid' })).toEqual({ padding: 'valid' });
    });
});
describe('convertTsToPythonic', function () {
    it('primitives', function () {
        expect(convertTsToPythonic(null)).toEqual(null);
        expect(convertTsToPythonic(true)).toEqual(true);
        expect(convertTsToPythonic(false)).toEqual(false);
        expect(convertTsToPythonic(4)).toEqual(4);
    });
    it('strings w/o name tags', function () {
        expect(convertTsToPythonic('abc')).toEqual('abc');
        expect(convertTsToPythonic('ABC')).toEqual('abc');
        expect(convertTsToPythonic('oneTwo')).toEqual('one_two');
        expect(convertTsToPythonic('OneTwo')).toEqual('one_two');
    });
    it('simple arrays', function () {
        expect(convertTsToPythonic([])).toEqual([]);
        expect(convertTsToPythonic([null])).toEqual([null]);
        expect(convertTsToPythonic(['oneTwo'])).toEqual(['one_two']);
        expect(convertTsToPythonic([null, true, false, 4, 'abc'])).toEqual([
            null, true, false, 4, 'abc'
        ]);
        expect(convertTsToPythonic([[[]]])).toEqual([[[]]]);
    });
    it('layer tuple (array) with key', function () {
        for (var _i = 0, _a = ['inboundNodes', 'inputLayers', 'outputLayers']; _i < _a.length; _i++) {
            var key = _a[_i];
            expect(convertTsToPythonic(['layerName', 'metaData', 0], key)).toEqual([
                'layerName', 'meta_data', 0
            ]);
        }
    });
    it('dictionary', function () {
        expect(convertTsToPythonic({})).toEqual({});
        expect(convertTsToPythonic({ key: null })).toEqual({ key: null });
        expect(convertTsToPythonic({ keyOne: 4 })).toEqual({ key_one: 4 });
        expect(convertTsToPythonic({ keyTwo: 'abcDef' })).toEqual({
            key_two: 'abc_def'
        });
        expect(convertTsToPythonic({ keyOne: true, keyTwo: false }))
            .toEqual({ key_one: true, key_two: false });
        expect(convertTsToPythonic({ name: 'layerName' })).toEqual({
            name: 'layerName'
        });
    });
    it('dictionary keys are passed down the stack', function () {
        var dict = {
            inboundNodes: ['DoNotChange_Me', 0, null]
        };
        expect(convertTsToPythonic(dict)).toEqual({
            inbound_nodes: ['DoNotChange_Me', 0, null]
        });
    });
    it('enum promotion', function () {
        expect(convertTsToPythonic({ mode: 'fanOut' })).toEqual({ mode: 'fan_out' });
        expect(convertTsToPythonic({ distribution: 'normal' })).toEqual({
            distribution: 'normal'
        });
        expect(convertTsToPythonic({ dataFormat: 'channelsLast' })).toEqual({
            data_format: 'channels_last'
        });
        expect(convertTsToPythonic({ padding: 'valid' })).toEqual({ padding: 'valid' });
    });
});
//# sourceMappingURL=serialization_utils_test.js.map
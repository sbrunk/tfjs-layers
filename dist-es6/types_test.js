import { SymbolicTensor } from './types';
describe('SymbolicTensor Test', function () {
    it('Correct dtype and shape properties', function () {
        var st1 = new SymbolicTensor('float32', [4, 6], null, [], {});
        expect(st1.dtype).toEqual('float32');
        expect(st1.shape).toEqual([4, 6]);
    });
    it('Correct names and ids', function () {
        var st1 = new SymbolicTensor('float32', [2, 2], null, [], {}, 'TestSymbolicTensor');
        var st2 = new SymbolicTensor('float32', [2, 2], null, [], {}, 'TestSymbolicTensor');
        expect(st1.name.indexOf('TestSymbolicTensor')).toEqual(0);
        expect(st2.name.indexOf('TestSymbolicTensor')).toEqual(0);
        expect(st1 === st2).toBe(false);
        expect(st1.id).toBeGreaterThanOrEqual(0);
        expect(st2.id).toBeGreaterThanOrEqual(0);
        expect(st1.id === st2.id).toBe(false);
    });
    it('Invalid tensor name leads to error', function () {
        expect(function () { return new SymbolicTensor('float32', [2, 2], null, [], {}, '!'); })
            .toThrowError();
    });
});
//# sourceMappingURL=types_test.js.map
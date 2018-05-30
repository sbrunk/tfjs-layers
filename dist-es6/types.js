var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import { doc } from '@tensorflow/tfjs-core';
import { getScopedTensorName, getUniqueTensorName } from './common';
var _nextUniqueTensorId = 0;
export function getNextUniqueTensorId() {
    return _nextUniqueTensorId++;
}
var SymbolicTensor = (function () {
    function SymbolicTensor(dtype, shape, sourceLayer, inputs, callArgs, name, outputTensorIndex) {
        this.dtype = dtype;
        this.shape = shape;
        this.sourceLayer = sourceLayer;
        this.inputs = inputs;
        this.callArgs = callArgs;
        this.outputTensorIndex = outputTensorIndex;
        this.id = getNextUniqueTensorId();
        if (name != null) {
            this.originalName = getScopedTensorName(name);
            this.name = getUniqueTensorName(this.originalName);
        }
    }
    SymbolicTensor = __decorate([
        doc({ heading: 'Models', 'subheading': 'Classes' })
    ], SymbolicTensor);
    return SymbolicTensor;
}());
export { SymbolicTensor };
//# sourceMappingURL=types.js.map
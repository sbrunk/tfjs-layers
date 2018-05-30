import { DataType, Scalar, serialization, Tensor } from '@tensorflow/tfjs-core';
import { Constraint } from '../constraints';
import { Initializer } from '../initializers';
import { Regularizer } from '../regularizers';
import { JsonDict, Kwargs, NamedTensorMap, RegularizerFn, Shape, SymbolicTensor } from '../types';
import { LayerVariable } from '../variables';
export declare type Op = (x: LayerVariable) => LayerVariable;
export interface InputSpecConfig {
    dtype?: DataType;
    shape?: Shape;
    ndim?: number;
    maxNDim?: number;
    minNDim?: number;
    axes?: {
        [axis: number]: number;
    };
}
export declare class InputSpec {
    dtype?: DataType;
    shape?: Shape;
    ndim?: number;
    maxNDim?: number;
    minNDim?: number;
    axes?: {
        [axis: number]: number;
    };
    constructor(config: InputSpecConfig);
}
export interface NodeConfig {
    outboundLayer: Layer;
    inboundLayers: Layer[];
    nodeIndices: number[];
    tensorIndices: number[];
    inputTensors: SymbolicTensor[];
    outputTensors: SymbolicTensor[];
    inputMasks: Tensor[];
    outputMasks: Tensor[];
    inputShapes: Shape | Shape[];
    outputShapes: Shape | Shape[];
}
export declare class Node {
    callArgs: Kwargs;
    outboundLayer: Layer;
    inboundLayers: Layer[];
    nodeIndices: number[];
    tensorIndices: number[];
    inputTensors: SymbolicTensor[];
    outputTensors: SymbolicTensor[];
    inputMasks: Tensor[];
    outputMasks: Tensor[];
    inputShapes: Shape | Shape[];
    outputShapes: Shape | Shape[];
    readonly id: number;
    constructor(config: NodeConfig, callArgs?: Kwargs);
    getConfig(): serialization.ConfigDict;
}
export interface LayerConfig {
    inputShape?: Shape;
    batchInputShape?: Shape;
    batchSize?: number;
    dtype?: DataType;
    name?: string;
    trainable?: boolean;
    updatable?: boolean;
    weights?: Tensor[];
    inputDType?: DataType;
}
export declare type CallHook = (inputs: Tensor | Tensor[], kwargs: Kwargs) => void;
export declare abstract class Layer extends serialization.Serializable {
    name: string;
    inputSpec: InputSpec[];
    supportsMasking: boolean;
    trainable: boolean;
    updatable: boolean;
    batchInputShape: Shape;
    dtype: DataType;
    initialWeights: Tensor[];
    inboundNodes: Node[];
    outboundNodes: Node[];
    activityRegularizer: Regularizer;
    protected _trainableWeights: LayerVariable[];
    private _nonTrainableWeights;
    private _losses;
    private _updates;
    private _built;
    private _callHook;
    private _addedWeightNames;
    readonly id: number;
    protected _stateful: boolean;
    constructor(config: LayerConfig);
    protected static nodeKey(layer: Layer, nodeIndex: number): string;
    private getNodeAtIndex(nodeIndex, attrName);
    getInputAt(nodeIndex: number): SymbolicTensor | SymbolicTensor[];
    getOutputAt(nodeIndex: number): SymbolicTensor | SymbolicTensor[];
    readonly input: SymbolicTensor | SymbolicTensor[];
    readonly output: SymbolicTensor | SymbolicTensor[];
    readonly losses: RegularizerFn[];
    calculateLosses(): Scalar[];
    readonly updates: Tensor[];
    built: boolean;
    trainableWeights: LayerVariable[];
    nonTrainableWeights: LayerVariable[];
    readonly weights: LayerVariable[];
    readonly stateful: boolean;
    protected assertInputCompatibility(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    protected invokeCallHook(inputs: Tensor | Tensor[], kwargs: Kwargs): void;
    setCallHook(callHook: CallHook): void;
    clearCallHook(): void;
    apply(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], kwargs?: Kwargs): Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[];
    build(inputShape: Shape | Shape[]): void;
    getWeights(trainableOnly?: boolean): Tensor[];
    setWeights(weights: Tensor[]): void;
    protected addWeight(name: string, shape: Shape, dtype?: DataType, initializer?: Initializer, regularizer?: Regularizer, trainable?: boolean, constraint?: Constraint): LayerVariable;
    addLoss(losses: RegularizerFn | RegularizerFn[]): void;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor | Tensor[];
    private addInboundNode(inputTensors, outputTensors, inputMasks, outputMasks, inputShapes, outputShapes, kwargs?);
    getConfig(): serialization.ConfigDict;
}
export interface InputLayerConfig {
    inputShape?: Shape;
    batchSize?: number;
    batchInputShape?: Shape;
    dtype?: DataType;
    sparse?: boolean;
    name?: string;
}
export declare class InputLayer extends Layer {
    static readonly className: string;
    sparse: boolean;
    constructor(config: InputLayerConfig);
    apply(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], kwargs?: Kwargs): Tensor | Tensor[] | SymbolicTensor;
    getConfig(): serialization.ConfigDict;
}
export interface InputConfig {
    shape?: Shape;
    batchShape?: Shape;
    name?: string;
    dtype?: DataType;
    sparse?: boolean;
}
export declare function Input(config: InputConfig): SymbolicTensor;
export interface ContainerConfig {
    inputs: SymbolicTensor | SymbolicTensor[];
    outputs: SymbolicTensor | SymbolicTensor[];
    name?: string;
}
export declare abstract class Container extends Layer {
    inputs: SymbolicTensor[];
    outputs: SymbolicTensor[];
    inputLayers: Layer[];
    inputLayersNodeIndices: number[];
    inputLayersTensorIndices: number[];
    outputLayers: Layer[];
    outputLayersNodeIndices: number[];
    outputLayersTensorIndices: number[];
    layers: Layer[];
    layersByDepth: {
        [depth: string]: Layer[];
    };
    nodesByDepth: {
        [depth: string]: Node[];
    };
    containerNodes: Set<string>;
    inputNames: string[];
    outputNames: string[];
    feedInputShapes: Shape[];
    protected internalInputShapes: Shape[];
    protected internalOutputShapes: Shape[];
    protected feedInputNames: string[];
    protected feedOutputNames: string[];
    constructor(config: ContainerConfig);
    readonly trainableWeights: LayerVariable[];
    readonly nonTrainableWeights: LayerVariable[];
    readonly weights: LayerVariable[];
    loadWeights(weightsJSON: JsonDict | NamedTensorMap, skipMismatch?: boolean, isNamedTensorMap?: boolean): void;
    private updatedConfig();
    toJSON(unused?: any, returnString?: boolean): string | JsonDict;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    protected runInternalGraph(inputs: Tensor[], masks?: Tensor[]): [Tensor[], Tensor[], Shape[]];
    private buildNodeConversionMap(layers);
    getLayer(name?: string, index?: number): Layer;
    calculateLosses(): Scalar[];
    getConfig(): serialization.ConfigDict;
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
    readonly stateful: boolean;
}
export declare function getSourceInputs(tensor: SymbolicTensor, layer?: Layer, nodeIndex?: number): SymbolicTensor[];
export declare function loadWeightsFromNamedTensorMap(weights: NamedTensorMap, layers: Layer[]): void;
export declare function loadWeightsFromJson(weightsJSON: JsonDict, layers: Layer[], skipMismatch?: boolean): void;

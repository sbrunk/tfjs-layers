import { Scalar, Tensor } from '@tensorflow/tfjs-core';
import { Model } from './engine/training';
export declare type UnresolvedLogs = {
    [key: string]: number | Scalar;
};
export declare type Logs = {
    [key: string]: number;
};
export declare type Params = {
    [key: string]: number | string | boolean | number[] | string[] | boolean[];
};
export declare abstract class Callback {
    validationData: Tensor | Tensor[];
    model: Model;
    params: Params;
    setParams(params: Params): void;
    setModel(model: Model): void;
    onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onTrainBegin(logs?: UnresolvedLogs): Promise<void>;
    onTrainEnd(logs?: UnresolvedLogs): Promise<void>;
}
export declare class CallbackList {
    callbacks: Callback[];
    queueLength: number;
    constructor(callbacks?: Callback[], queueLength?: number);
    append(callback: Callback): void;
    setParams(params: Params): void;
    setModel(model: Model): void;
    onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onTrainBegin(logs?: UnresolvedLogs): Promise<void>;
    onTrainEnd(logs?: UnresolvedLogs): Promise<void>;
}
export declare class BaseLogger extends Callback {
    private seen;
    private totals;
    constructor();
    onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
}
export declare function resolveScalarsInLogs(logs: UnresolvedLogs): Promise<void>;
export declare function disposeTensorsInLogs(logs: UnresolvedLogs): void;
export declare class History extends Callback {
    epoch: number[];
    history: {
        [key: string]: Array<number | Tensor>;
    };
    onTrainBegin(logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    syncData(): Promise<void>;
}
export interface CustomCallbackConfig {
    onTrainBegin?: (logs?: Logs) => Promise<void>;
    onTrainEnd?: (logs?: Logs) => Promise<void>;
    onEpochBegin?: (epoch: number, logs?: Logs) => Promise<void>;
    onEpochEnd?: (epoch: number, logs?: Logs) => Promise<void>;
    onBatchBegin?: (batch: number, logs?: Logs) => Promise<void>;
    onBatchEnd?: (batch: number, logs?: Logs) => Promise<void>;
}
export declare class CustomCallback extends Callback {
    protected readonly trainBegin: (logs?: Logs) => Promise<void>;
    protected readonly trainEnd: (logs?: Logs) => Promise<void>;
    protected readonly epochBegin: (epoch: number, logs?: Logs) => Promise<void>;
    protected readonly epochEnd: (epoch: number, logs?: Logs) => Promise<void>;
    protected readonly batchBegin: (batch: number, logs?: Logs) => Promise<void>;
    protected readonly batchEnd: (batch: number, logs?: Logs) => Promise<void>;
    constructor(config: CustomCallbackConfig);
    onEpochBegin(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onEpochEnd(epoch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchBegin(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onBatchEnd(batch: number, logs?: UnresolvedLogs): Promise<void>;
    onTrainBegin(logs?: UnresolvedLogs): Promise<void>;
    onTrainEnd(logs?: UnresolvedLogs): Promise<void>;
}
export declare function standardizeCallbacks(callbacks: Callback | Callback[] | CustomCallbackConfig | CustomCallbackConfig[]): Callback[];


using FLAB;
using Sofia.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Sofia
{
    public class Optimizer
    {
        protected NeuralNetwork _network;
        protected double _naturalEpsilon;

        protected Dictionary<string, Matrix> _dW;
        protected Dictionary<string, Vector> _db;
        protected Dictionary<string, Matrix> _gradient;
        protected Dictionary<string, Tensor> _gradient_t;
        protected float _weightDecay;

        private float _startAlpha, _endAlpha;
        protected float _alpha;

        protected BatchingUnit _batchingUnit;
        protected bool _asyncMode;

        protected Optimizer(NeuralNetwork p_network, float p_weightDecay = 0)
        {
            _network = p_network;
            _naturalEpsilon = 1e-3;
            _weightDecay = p_weightDecay;
            _batchingUnit = new BatchingUnit();
            _batchingUnit.Init();
            _asyncMode = false;

            _dW = new Dictionary<string, Matrix>();
            _db = new Dictionary<string, Vector>();

            foreach (Connection c in _network.Connections.Values)
            {
                _dW[c.Id] = Matrix.Zero(c.OutDim, c.InDim);
                _db[c.Id] = Vector.Zero(c.OutDim);
            }
        }

        public void InitBatchMode(int p_size)
        {
            _batchingUnit.Init(p_size);
        }

        public void InitAsynchMode(bool p_value)
        {
            _asyncMode = p_value;
        }

        public virtual void Dispose()
        {
            foreach (Vector db in _db.Values)
            {
                Vector.Release(db);
            }
            foreach (Matrix dW in _dW.Values)
            {
                Matrix.Release(dW);
            }
            _network.Dispose();
        }

        public virtual float Train(Vector p_input, Vector p_target)
        {
            return 0;
        }

        public virtual float Train(Tensor p_input, Tensor p_target)
        {
            return 0;
        }

        protected virtual void CalcWeightUpdate(Connection p_connection, bool p_tensor)
        {
            return;
        }

        protected float CalcMse(Vector p_target)
        {
            Vector e = p_target - _network.Output;

            float mse = 0;
            // calc MSE
            for (int i = 0; i < _network.Output.Size; i++)
            {
                mse += (float)Math.Pow(e[i], 2);
            }

            Vector.Release(e);

            return mse;
        }

        protected float CalcMse(Tensor p_target)
        {
            Tensor te = (p_target - _network.OutputT);
            te.Apply(Math.Pow, 2);
            Vector e = Vector.Build(_network.Output.Size, te.ReduceSum(0).Buffer);

            float mse = 0;
            // calc MSE
            for (int i = 0; i < e.Size; i++)
            {
                mse += e[i];
            }

            return mse;
        }

        protected void Update(Vector p_error)
        {
            CalcGradient(p_error);
            _batchingUnit.Update();

            foreach (Connection c in _network.Connections.Values)
            {
                UpdateConnection(c, false);
            }
        }

        protected void Update(Tensor p_error)
        {
            CalcGradient(p_error);

            foreach (Connection c in _network.Connections.Values)
            {
                UpdateConnection(c, true);
            }
        }

        private void UpdateConnection(Connection p_connection, bool p_tensor)
        {
            if (p_connection.Trainable)
            {
                CalcWeightUpdate(p_connection, p_tensor);
                if (_weightDecay != 0) WeightDecay(p_connection, p_tensor);

                if (_batchingUnit.IsBatchFinished && !_asyncMode) { 
                    p_connection.Update(_dW[p_connection.Id]);
                    p_connection.OutGroup.UpdateBias(_db[p_connection.Id]);

                    if (_batchingUnit.IsActive) {
                        _dW[p_connection.Id].Fill(0f);
                        _db[p_connection.Id].Fill(0f);
                    }                    
                }
            }
        }

        public void AsyncUpdate()
        {
            if (_asyncMode)
            {
                foreach (Connection c in _network.Connections.Values)
                {
                    if (c.Trainable)
                    {
                        c.Update(_dW[c.Id]);
                        c.OutGroup.UpdateBias(_db[c.Id]);

                        _dW[c.Id].Fill(0f);
                        _db[c.Id].Fill(0f);
                    }
                }
            }
        }

        protected void WeightDecay(Connection p_connection, bool p_tensor)
        {
            if (_weightDecay > 0)
            {
                if (p_tensor)
                {
                    p_connection.Weights *= (float)Math.Pow((1 - _weightDecay), p_connection.OutGroup.Batch);
                }
                else
                {
                    Matrix w = p_connection.Weights * (1 - _weightDecay);
                    Matrix.Release(p_connection.Weights);
                    p_connection.Weights = w;
                }
            }
        }

        protected void CalcGradient(Vector p_error)
        {
            _gradient = _network.CalcGradient(p_error);
        }

        protected void CalcGradient(Tensor p_error)
        {
            _gradient_t = _network.CalcGradient(p_error);
        }

        public void InitAlpha(float p_startAlpha, float p_endAlpha)
        {
            _startAlpha = p_startAlpha;
            _endAlpha = p_endAlpha;
            _alpha = _startAlpha;
        }

        public void UpdateAlpha(float p_f)
        {
            _alpha = _startAlpha + (_endAlpha - _startAlpha) * p_f;
        }

        public float Alpha
        {
            set { _alpha = value; }
            get { return _alpha; }
        }
    }
}

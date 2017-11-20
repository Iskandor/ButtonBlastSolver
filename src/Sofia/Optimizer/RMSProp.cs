using FLAB;
using System;
using System.Collections.Generic;

namespace Sofia
{
    public class RMSProp : Optimizer
    {
        private float _cacheDecay;
        private float _epsilon;

        private Dictionary<string, Tensor> _eps_t;
        private Dictionary<string, Matrix> _gradientCache;
        private Dictionary<string, Tensor> _gradientCache_t;

        public RMSProp(NeuralNetwork p_network, float p_cacheDecay = 0.9f, float p_epsilon = 1e-8f) : base(p_network)
        {
            int nRows;
            int nCols;

            _cacheDecay = p_cacheDecay;
            _epsilon = p_epsilon;
            _gradientCache = new Dictionary<string, Matrix>();
            _gradientCache_t = new Dictionary<string, Tensor>();

            _eps_t = new Dictionary<string, Tensor>();

            foreach (Connection c in _network.Connections.Values)
            {
                if (c.Trainable)
                {
                    nRows = c.OutDim;
                    nCols = c.InDim;
                    _gradientCache[c.Id] = Matrix.Zero(nRows, nCols);
                }
            }
        }

        override public void Dispose()
        {
            base.Dispose();
            foreach (Connection c in _network.Connections.Values)
            {
                Matrix.Release(_gradientCache[c.Id]);
            }
        }

        override public float Train(Vector p_input, Vector p_target)
        {
            float mse = 0;
            // forward activation phase
            _network.Activate(p_input);

            // backward training phase
            Vector error = p_target - _network.Output;

            mse = CalcMse(p_target);
            Update(error);

            Vector.Release(error);

            return mse;
        }

        override public float Train(Tensor p_input, Tensor p_target)
        {
            float mse = 0;
            // forward activation phase
            _network.Activate(p_input);

            // backward training phase
            Tensor error = p_target - _network.OutputT;

            mse = CalcMse(p_target);
            Update(error);

            return mse;
        }

        override protected void CalcWeightUpdate(Connection p_connection, bool p_tensor)
        {
            string id = p_connection.Id;

            if (p_tensor)
            {
                if (!_gradientCache_t.ContainsKey(id))
                {
                    int batch = (int)p_connection.GradientT.GetShape(0);
                    _gradientCache_t[id] = new Tensor(batch, p_connection.OutDim, p_connection.InDim);
                    //_eps_t[id] = new Tensor(_eps[id], batch);
                }

                Tensor g = new Tensor(_gradient_t[id]);

                g.Apply(Math.Pow, 2);

                _gradientCache_t[id].RecurrentSum(_cacheDecay, (1 - _cacheDecay) * g);

                g = new Tensor(_gradientCache_t[id]);
                g.Apply(Math.Sqrt);

                Tensor gdiv = g + _eps_t[id];
                gdiv.Inv();
                _gradient_t[id].Dot(gdiv);

                p_connection.Update(_alpha * _gradient_t[id]);
                p_connection.OutGroup.UpdateBias(_alpha * p_connection.OutGroup.DeltaT);
            }
            else
            {
                Matrix gc = Matrix.RMS_gradientCache(_cacheDecay, _gradientCache[id], _gradient[id]);
                Matrix.Release(_gradientCache[id]);
                _gradientCache[id] =  gc;

                if (_batchingUnit.IsActive)
                {
                    Matrix gu = Matrix.RMS_gradientUpdate(_alpha, _gradientCache[id], _gradient[id], _epsilon);
                    Matrix dW = _dW[id] + gu;
                    Matrix.Release(gu);
                    Matrix.Release(_dW[id]);
                    _dW[id] = dW;

                    Vector bu = _alpha * p_connection.OutGroup.Delta;
                    Vector db = _db[id] + bu;
                    Vector.Release(bu);
                    Vector.Release(_db[id]);
                    _db[id] = db;
                }
                else
                {
                    Matrix.Release(_dW[id]);
                    _dW[id] = Matrix.RMS_gradientUpdate(_alpha, _gradientCache[id], _gradient[id], _epsilon);

                    Vector.Release(_db[id]);
                    _db[id] = _alpha * p_connection.OutGroup.Delta;
                }
            }
        }

    }
}

using FLAB;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Sofia
{
    public class ADAM : Optimizer
    {
        private float _beta1, _beta2, _epsilon;
        //private Dictionary<string, Matrix> _eps;
        private Dictionary<string, Matrix> _m;
        private Dictionary<string, Matrix> _v;
        private Dictionary<string, Tensor> _eps_t;
        private Dictionary<string, Tensor> _m_t;
        private Dictionary<string, Tensor> _v_t;

        public ADAM(NeuralNetwork p_network, float p_beta1 = .9f, float p_beta2 = .999f, float p_epsilon = 1e-8f) : base(p_network)
        {
            _beta1 = p_beta1;
            _beta2 = p_beta2;
            _epsilon = p_epsilon;

            int nRows;
            int nCols;

            _m = new Dictionary<string, Matrix>();
            _v = new Dictionary<string, Matrix>();
            //_eps = new Dictionary<string, Matrix>();
            _m_t = new Dictionary<string, Tensor>();
            _v_t = new Dictionary<string, Tensor>();
            _eps_t = new Dictionary<string, Tensor>();

            foreach (Connection c in _network.Connections.Values)
            {
                if (c.Trainable)
                {
                    nRows = c.OutGroup.Dim;
                    nCols = c.InGroup.Dim;
                    _m[c.Id] = Matrix.Zero(nRows, nCols);
                    _v[c.Id] = Matrix.Zero(nRows, nCols);
                    //_eps[c.Id] = Matrix.Value(nRows, nCols, p_epsilon);
                }
            }
        }

        override public void Dispose()
        {
            base.Dispose();
            foreach (Connection c in _network.Connections.Values)
            {
                Matrix.Release(_m[c.Id]);
                Matrix.Release(_v[c.Id]);
                //BasePool.Instance.Release(_eps[c.Id]);
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

            Stopwatch watch = Stopwatch.StartNew();

            // forward activation phase
            _network.Activate(p_input);

            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds);

            // backward training phase
            Tensor error = p_target - _network.OutputT;

            mse = CalcMse(p_target);

            watch.Start();

            Update(error);

            watch.Stop();
            Console.WriteLine(watch.ElapsedMilliseconds);

            return mse;
        }

        override protected void CalcWeightUpdate(Connection p_connection, bool p_tensor)
        {
            /*
            m = beta1*m + (1-beta1)*dx
            v = beta2*v + (1-beta2)*(dx**2)
            x += - learning_rate * m / (np.sqrt(v) + eps)
            */

            string id = p_connection.Id;

            if (p_tensor)
            {
                if (!_m_t.ContainsKey(id))
                {
                    int batch = (int)p_connection.GradientT.GetShape(0);
                    _m_t[id] = new Tensor(batch, p_connection.OutDim, p_connection.InDim);
                    _v_t[id] = new Tensor(batch, p_connection.OutDim, p_connection.InDim);
                    //_eps_t[id] = new Tensor(_eps[id], batch);
                }

                Tensor g = new Tensor(_gradient_t[id]);
                g.Apply(Math.Pow, 2);
                
                _m_t[id].RecurrentSum(_beta1, (1 - _beta1) * _gradient_t[id]);
                _v_t[id].RecurrentSum(_beta2, (1 - _beta2) * g);

                Tensor v = new Tensor(_v_t[id]);
                v.Apply(Math.Sqrt);

                Tensor gdiv = (v + _eps_t[id]);
                gdiv.Inv();

                Tensor m = new Tensor(_m_t[id]);
                m.Dot(gdiv);

                p_connection.Update(_alpha * m);
                p_connection.OutGroup.UpdateBias(_alpha * p_connection.OutGroup.DeltaT);
            }
            else
            {
                Matrix m = Matrix.ADAM_mCache(_beta1, _m[id], _gradient[id]);
                Matrix.Release(_m[id]);
                _m[id] = m;

                Matrix v = Matrix.ADAM_vCache(_beta2, _v[id], _gradient[id]);
                Matrix.Release(_v[id]);
                _v[id] = v;

                if (_batchingUnit.IsActive)
                {
                    Matrix gu = Matrix.ADAM_gradientUpdate(_alpha, _v[id], _m[id], _epsilon);
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
                    _dW[id] = Matrix.ADAM_gradientUpdate(_alpha, _v[id], _m[id], _epsilon);

                    Vector.Release(_db[id]);
                    _db[id] = _alpha * p_connection.OutGroup.Delta;
                }

            }
        }
    }
}

using FLAB;
using Sofia.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Sofia
{
    public class BackProp : Optimizer
    {
        private Dictionary<string, Matrix> _v;
        private Dictionary<string, Matrix> _v_prev;
        private Dictionary<string, Tensor> _v_t;

        private float _momentum;
        private bool _nesterov;

        public BackProp(NeuralNetwork p_network, float p_weightDecay = 0, float p_momentum = 0, bool p_nesterov = false) : base(p_network, p_weightDecay)
        {
            _momentum = p_momentum;
            _nesterov = p_nesterov;

            int nRows;
            int nCols;

            _v = new Dictionary<string, Matrix>();
            _v_prev = new Dictionary<string, Matrix>();
            _v_t = new Dictionary<string, Tensor>();

            foreach (Connection c in _network.Connections.Values)
            {
                nRows = c.OutGroup.Dim;
                nCols = c.InGroup.Dim;
                _v[c.Id] = Matrix.Zero(nRows, nCols);
                _v_prev[c.Id] = null;
            }
        }

        override public void Dispose()
        {
            base.Dispose();
            foreach (Connection c in _network.Connections.Values)
            {
                Matrix.Release(_v[c.Id]);
                Matrix.Release(_v_prev[c.Id]);
            }
        }

        override public float Train(Vector p_input, Vector p_target)
        {
            float mse = 0;
            // forward activation phase
            _network.Activate(p_input);
            mse = CalcMse(p_target);

            // backward training phase
            Vector error = p_target - _network.Output;
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
            /*
            v_prev = v # back this up
            v = mu * v - learning_rate * dx # velocity update stays the same
            x += -mu * v_prev + (1 + mu) * v # position update changes form
            */
            string id = p_connection.Id;

            if (p_tensor)
            {
                if (!_v_t.ContainsKey(id))
                {
                    int batch = (int)p_connection.GradientT.GetShape(0);
                    _v_t[id] = new Tensor(batch, p_connection.OutDim, p_connection.InDim);
                }

                Tensor v_prev = null;

                if (_momentum > 0)
                {
                    _v_t[id].RecurrentSum(_momentum, _alpha * _gradient_t[id]);
                }
                else
                {
                    _v_t[id] = _alpha * _gradient_t[id];
                }

                if (_nesterov)
                {
                    if (v_prev != null)
                    {
                        p_connection.Update(-_momentum * v_prev + (1 + _momentum) * _v_t[id]);
                    }
                    else
                    {
                        p_connection.Update((1 + _momentum) * _v_t[id]);
                    }
                    
                    v_prev = new Tensor(_v_t[id]);
                    v_prev.RShift(0);
                }
                else
                {
                    p_connection.Update(_v_t[id]);
                }

                p_connection.OutGroup.UpdateBias(_alpha * p_connection.OutGroup.DeltaT);
            }
            else
            {
                if (_momentum > 0)
                {
                    Matrix dv1 = _momentum * _v[id];
                    Matrix dv2 = _alpha * _gradient[id];
                    Matrix dv = dv1 + dv2;
                    
                    Matrix.Release(dv1);
                    Matrix.Release(dv2);
                    Matrix.Release(_v[id]);

                    _v[id] = dv;
                    p_connection.Update(_v[id]);
                }
                else
                {
                    Matrix dv = _alpha * _gradient[id];
                    p_connection.Update(dv);
                    Matrix.Release(dv);
                }

                if (_momentum > 0 && _nesterov)
                {
                    if (_v_prev[id] != null)
                    {
                        Matrix dw1 = -_momentum * _v_prev[id];
                        Matrix dw2 = (1 + _momentum) * _v[id];
                        Matrix dw = dw1 + dw2;
                        p_connection.Update(dw);
                        Matrix.Release(dw1);
                        Matrix.Release(dw2);
                        Matrix.Release(dw);
                    }
                    else
                    {
                        Matrix dw = (1 + _momentum) * _v[id];
                        p_connection.Update(dw);
                        Matrix.Release(dw);
                    }

                    Matrix.Release(_v_prev[id]);
                    _v_prev[id] = Matrix.Copy(_v[id]);
                }

                Vector db = _alpha * p_connection.OutGroup.Delta;

                p_connection.OutGroup.UpdateBias(db);
                Vector.Release(db);
            }
        }
    }
}

using System;
using System.Collections.Generic;

namespace FLAB
{
    class BasePool
    {
        private int _matrixCounter;
        private int _vectorCounter;
        private Dictionary<long, Stack<Matrix>> _poolMatrix;
        private Dictionary<long, Stack<Vector>> _poolVector;
        private static BasePool _instance;

        public static bool StrongControl = false;

        public static BasePool Instance
        {
            get {
                if (_instance == null)
                    _instance = new BasePool();
                return _instance;
            }
        }

        private BasePool()
        {
            _poolMatrix = new Dictionary<long, Stack<Matrix>>();
            _poolVector = new Dictionary<long, Stack<Vector>>();
            _matrixCounter = 0;
            _vectorCounter = 0;
        }

        public Matrix Get(int p_rows, int p_cols)
        {
            Matrix result = null;

            long size = p_rows * p_cols;

            if (!_poolMatrix.ContainsKey(size))
            {
                _poolMatrix[size] = new Stack<Matrix>();

                for(int i = 0; i < 10; i++)
                {
                    _poolMatrix[size].Push(new Matrix(p_rows, p_cols));
                } 
            }

            if (_poolMatrix[size].Count == 0)
            {
                _matrixCounter++;
                result = new Matrix(p_rows, p_cols);
            }
            else
            {
                _matrixCounter++;
                result = _poolMatrix[size].Pop();
                result.Rows = p_rows;
                result.Cols = p_cols;
            }

            return result;
        }

        public Vector Get(int p_size)
        {
            Vector result = null;

            long size = p_size;

            if (!_poolVector.ContainsKey(size))
            {
                _poolVector[size] = new Stack<Vector>();
                for (int i = 0; i < 100; i++)
                {                    
                    _poolVector[size].Push(new Vector(p_size));
                }
            }

            if (_poolVector[size].Count == 0)
            {
                _vectorCounter++;
                result = new Vector(p_size);
            }
            else
            {
                _vectorCounter++;
                result = _poolVector[size].Pop();
                result.Rows = p_size;
                result.Cols = 1;
            }

            return result;
        }

        public void Release(Matrix p_item)
        {
            if (p_item != null)
            {
                _matrixCounter--;
                long size = p_item.Rows * p_item.Cols;

                if (StrongControl)
                {
                    foreach (Matrix m in _poolMatrix[size])
                    {
                        if (m.Id.Equals(p_item.Id))
                        {
                            Console.WriteLine("Releasing matrix twice");
                        }
                    }
                }

                _poolMatrix[size].Push(p_item);
            }
        }

        public void Release(Vector p_item)
        {
            if (p_item != null)
            {
                _vectorCounter--;
                long size = p_item.Size;

                if (StrongControl)
                {
                    foreach (Vector v in _poolVector[size])
                    {
                        if (v.Id.Equals(p_item.Id))
                        {
                            Console.WriteLine("Releasing vector twice");
                        }
                    }
                }

                _poolVector[size].Push(p_item);
            }
        }

        public void Check()
        {
            Console.WriteLine("Pool check > live matrices: " + _matrixCounter + " live vectors: " + _vectorCounter);
            Console.WriteLine("Pool check > Vectors created " + Vector.DEBUG_Counter);
            Console.WriteLine("Pool check > Matrices created " + Matrix.DEBUG_Counter);
            Vector.DEBUG_Counter = Matrix.DEBUG_Counter = 0;
        }

    }
}

using Sofia;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace pFLAB
{
    public enum INIT
    {
        ZERO = 0,
        IDENTITY = 1,
        ONES = 1,
        VALUE = 2,
        RANDOM = 3
    };

    public class Tensor
    {
        protected int _rank;
        protected long[] _shape;
        protected float[] _arr;

        private long[] _offset;
        private long _size;

        public Tensor(params long[] p_shape)
        {
            Init(null, p_shape);
        }

        public Tensor(float[] p_data, params long[] p_shape)
        {
            Init(p_data, p_shape);
            _arr = p_data;
        }

        public Tensor(Tensor p_copy)
        {
            _rank = p_copy._rank;
            _size = p_copy._size;
            _arr = new float[_size];
            _shape = new long[_rank];
            _offset = new long[_rank];
            Array.Copy(p_copy._shape, _shape, p_copy._shape.GetLength(0));
            Array.Copy(p_copy._arr, _arr, p_copy._size);
            Array.Copy(p_copy._offset, _offset, p_copy._rank);
        }

        public Tensor(Vector[] p_list)
        {
            _rank = 2;
            _shape = new long[_rank];

            _shape[0] = p_list.GetLength(0);
            _shape[1] = p_list[0].Size;

            Init(null, _shape);

            int index = 0;

            foreach (Vector v in p_list)
            {
                Array.Copy(v.Arr, 0, _arr, index, v.Size);
                index += v.Size;
            }
        }

        public Tensor(Vector p_vector, int p_newDim)
        {
            _rank = 2;
            _shape = new long[_rank];

            _shape[0] = p_newDim;
            _shape[1] = p_vector.Size;

            Init(null, _shape);

            int index = 0;

            for (int i = 0; i < p_newDim; i++)
            {
                Array.Copy(p_vector.Arr, 0, _arr, index, p_vector.Size);
                index += p_vector.Size;
            }
        }

        public Tensor(Matrix p_matrix, int p_newDim)
        {
            _rank = 3;
            _shape = new long[_rank];

            _shape[0] = p_newDim;
            _shape[1] = p_matrix.Rows;
            _shape[2] = p_matrix.Cols;

            Init(null, _shape);

            int index = 0;

            for (int i = 0; i < p_newDim; i++)
            {
                Array.Copy(p_matrix.Arr, 0, _arr, index, p_matrix.Rows * p_matrix.Cols);
                index += p_matrix.Rows * p_matrix.Cols;
            }
        }

        private void Init(float[] p_data, params long[] p_shape)
        {
            _shape = p_shape;
            _rank = p_shape.GetLength(0);
            _offset = new long[_rank];
            _offset[_rank - 1] = 1;

            _size = 1;

            for (int i = 0; i < _rank; i++)
            {
                _size *= _shape[i];
            }

            for (int i = 0; i < _rank - 1; i++)
            {
                _offset[i] = 1;

                for (int j = _rank - 1; j > i; j--)
                {
                    _offset[i] *= _shape[j];
                }

            }

            if (p_data == null)
            {
                _arr = new float[_size];
            }
            else
            {
                _arr = p_data;
            }
        }

        public void Apply(Func<double, double> p_func)
        {
            for (int i = 0; i < _size; i++)
            {
                _arr[i] = (float)p_func(_arr[i]);
            }
        }

        public void Apply(Func<double, double, double> p_func, double p_y)
        {
            for (int i = 0; i < _size; i++)
            {
                _arr[i] = (float)p_func(_arr[i], p_y);
            }
        }

        public void Fill(float p_value)
        {
            for (int i = 0; i < _size; i++)
            {
                _arr[i] = p_value;
            }
        }

        public long GetShape(int p_index)
        {
            return _shape[p_index];
        }

        /* velmi draha operacia, nepouzivat v operatoroch
        public float this[params int[] p_index]
        {
            get
            {
                long index = 0;

                for(int i = 0; i < p_index.GetLength(0); i++)
                {
                    index += p_index[i] * _offset[i];
                }

                return _arr[index];
            }

            set
            {
                long index = 0;

                for (int i = 0; i < p_index.GetLength(0); i++)
                {
                    index += p_index[i] * _offset[i];
                }

                _arr[index] = value;
            }
        }
        */


        public static Tensor operator *(float p_const1, Tensor p_tensor2)
        {
            Tensor result = new Tensor(p_tensor2);

            for (int i = 0; i < result._size; i++)
            {
                result._arr[i] = p_tensor2._arr[i] * p_const1;
            }

            return result;
        }

        public static Tensor operator *(Tensor p_tensor1, float p_const2)
        {
            return p_const2 * p_tensor1;
        }

        public static Tensor operator *(Tensor p_tensor1, Tensor p_tensor2)
        {
            Tensor res = null;

            /* zatial pre 3D * 2D TODO prepisat na nD a n-1D */
            if (p_tensor1._rank == p_tensor2._rank + 1)
            {
                res = new Tensor(new long[] { p_tensor1.GetShape(0), p_tensor1.GetShape(1) });

                for (int d = 0; d < p_tensor1.GetShape(0); d++)
                {
                    for (int i = 0; i < p_tensor1.GetShape(1); i++)
                    {
                        for (int j = 0; j < p_tensor1.GetShape(2); j++)
                        {
                            res._arr[d * res._offset[0] + i] += p_tensor1._arr[d * p_tensor1._offset[0] + i * p_tensor1._offset[1] + j] * p_tensor2._arr[d * p_tensor2._offset[0] + j];
                        }
                    }
                }
            }

            /* zatial pre 2D * 2D (tensor product) TODO prepisat na nD a nD */
            if (p_tensor1._rank == p_tensor2._rank)
            {
                res = new Tensor(new long[] { p_tensor1.GetShape(0), p_tensor1.GetShape(1), p_tensor2.GetShape(1) });

                for (int d = 0; d < p_tensor1.GetShape(0); d++)
                {
                    for (int i = 0; i < p_tensor1.GetShape(1); i++)
                    {
                        for (int j = 0; j < p_tensor2.GetShape(1); j++)
                        {
                            res._arr[d * res._offset[0] + i * res._offset[1] + j] += p_tensor1._arr[d * p_tensor1._offset[0] + i] * p_tensor2._arr[d * p_tensor2._offset[0] + j];
                        }
                    }
                }
            }

            return res;
        }

        public static Tensor operator +(Tensor p_tensor1, Tensor p_tensor2)
        {
            Tensor res = new Tensor(p_tensor1.Shape);

            for (int i = 0; i < p_tensor1._size; i++)
            {
                res._arr[i] = p_tensor1._arr[i] + p_tensor2._arr[i];
            }

            return res;
        }

        public static Tensor operator -(Tensor p_tensor1, Tensor p_tensor2)
        {
            Tensor res = new Tensor(p_tensor1.Shape);

            for (int i = 0; i < p_tensor1._size; i++)
            {
                res._arr[i] = p_tensor1._arr[i] - p_tensor2._arr[i];
            }

            return res;
        }

        public static bool operator ==(Tensor p_tensor1, Tensor p_tensor2)
        {
            if ((object)p_tensor1 == null && (object)p_tensor2 == null)
            {
                return true;
            }

            bool result = (object)p_tensor1 != null && (object)p_tensor2 != null;

            if (result)
            {
                result = result && p_tensor1._rank == p_tensor2._rank;
            }

            if (result)
            {
                for (int i = 0; i < p_tensor1._rank; i++)
                {
                    result = result && p_tensor1._shape[i] == p_tensor2._shape[i];
                }
            }

            if (result)
            {
                for (int i = 0; i < p_tensor1._size; i++)
                {
                    result = result && p_tensor1._arr[i] == p_tensor2._arr[i];
                }
            }

            return result;
        }

        public static bool operator !=(Tensor p_tensor1, Tensor p_tensor2)
        {
            return !(p_tensor1 == p_tensor2);
        }

        public bool Equals(Tensor p_tensor)
        {
            return this == p_tensor;
        }

        /* 2D -> 3D v diag. forme */
        public static Tensor Diag(Tensor p_tensor)
        {
            long b = p_tensor.GetShape(0);
            long d = p_tensor.GetShape(1);

            Tensor result = new Tensor(b, d, d);

            for (int i = 0; i < b; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    result._arr[i * result._offset[0] + j * result._offset[1] + j] = p_tensor._arr[i * p_tensor._offset[0] + j];
                }
            }

            return result;
        }

        /* zatial 2D = Matrix.T TODO zovseobecnit na perm argument*/
        public Tensor T()
        {
            Tensor res = new Tensor(_shape[1], _shape[0]);

            for (int i = 0; i < _shape[0]; i++)
            {
                for (int j = 0; j < _shape[1]; j++)
                {
                    res._arr[j * res._shape[1] + i] = _arr[i * _shape[1] + j];
                }
            }

            return res;
        }

        /* kontrakcia z na nD, nD-1 so Sum operatorom TODO prerobit na arg operator? */
        public Tensor ReduceSum(int p_dim)
        {
            long[] newShape = new long[_rank - 1];

            for (int i = 0; i < _rank; i++)
            {
                if (i < p_dim)
                {
                    newShape[i] = _shape[i];
                }
                else if (i > p_dim)
                {
                    newShape[i - 1] = _shape[i];
                }
            }

            Tensor res = new Tensor(newShape);

            for (int i = 0; i < _size; i++)
            {
                int index = (int)(i % res._size);
                res._arr[index] += _arr[i];
            }

            return res;
        }

        /* nD cez dim 0 */
        public void RecurrentSum(float p_ratio, Tensor p_tensor)
        {
            int shapes = 1;

            for (int i = 1; i < _rank; i++)
            {
                shapes *= (int)_shape[i];
            }

            float[] tmp = new float[shapes];

            for (int i = 0; i < (int)_shape[0]; i++)
            {
                int b = i > 0 ? i - 1 : (int)(_shape[0] - 1);

                for (int j = 0; j < shapes; j++)
                {
                    _arr[i * _offset[0] + j] = p_ratio * _arr[b * _offset[0] + j] + p_tensor._arr[i * _offset[0] + j];
                }
            }
        }

        // pre ine dimenzie nefunguje
        public void RShift(int p_dim = 0, bool p_rotation = true)
        {
            int shapes = 1;

            for (int i = 0; i < _rank; i++)
            {
                if (i != p_dim)
                {
                    shapes *= (int)_shape[i];
                }
            }

            float[] tmp = new float[shapes];

            for (int i = 0; i < shapes; i++)
            {
                tmp[i] = _arr[(_shape[p_dim] * _offset[p_dim] - 1) - i];
                for (int j = (int)_shape[p_dim]; j > 1; j--)
                {
                    _arr[(j * _offset[p_dim] - 1) - i] = _arr[((j - 1) * _offset[p_dim] - 1) - i];
                }
                if (p_rotation)
                {
                    _arr[(_offset[p_dim] - 1 - i)] = tmp[i];
                }
            }
        }

        // pre ine dimenzie nefunguje
        public void LShift(int p_dim = 0, bool p_rotation = true)
        {
            int shapes = 1;

            for (int i = 0; i < _rank; i++)
            {
                if (i != p_dim)
                {
                    shapes *= (int)_shape[i];
                }
            }

            float[] tmp = new float[shapes];

            for (int i = 1; i <= shapes; i++)
            {
                tmp[i - 1] = _arr[i - 1];
                for (int j = 0; j < (int)_shape[p_dim] - 1; j++)
                {
                    _arr[(j * _offset[p_dim] - 1) + i] = _arr[((j + 1) * _offset[p_dim] - 1) + i];
                }
                if (p_rotation)
                {
                    _arr[(((int)_shape[p_dim] - 1) * _offset[p_dim] - 1) + i] = tmp[i - 1];
                }
            }
        }

        public void Inv()
        {
            for (int i = 0; i < _size; i++)
            {
                _arr[i] = 1f / _arr[i];
            }
        }

        public void Dot(Tensor p_tensor)
        {
            for (int i = 0; i < _size; i++)
            {
                _arr[i] *= p_tensor._arr[i];
            }
        }

        override public string ToString()
        {
            string res = string.Empty;

            for (int i = 0; i < _size; i++)
            {
                res += _arr[i] + " ";
            }

            res += "\n";

            return res;
        }

        public long[] Shape
        {
            get { return _shape; }
        }

        public float[] Buffer
        {
            get { return _arr; }
        }
    }

    public class Matrix : Base
    {
        public Matrix(int p_rows = 0, int p_cols = 0, INIT p_init = INIT.ZERO, float p_value = 0) : base(p_rows, p_cols)
        {
            Init(p_init, p_value);
        }

        public Matrix(int p_rows, int p_cols, float[] p_data) : base(p_rows, p_cols, p_data)
        {
        }

        public Matrix(int p_rows, int p_cols, List<float> inputs) : base(p_rows, p_cols, inputs) {
        }

        public Matrix(Matrix p_copy) : base(p_copy)
        {

        }

        new void Init(INIT p_init, float p_value)
        {
            switch (p_init)
            {
                case INIT.ZERO:
                    Fill(0);
                    break;
                case INIT.IDENTITY:
                    Fill(0);
                    for (int i = 0; i < _rows; i++)
                    {
                        _arr[i * _cols + i] = 1;
                    }
                    break;
                case INIT.VALUE:
                    Fill(p_value);
                    break;
                case INIT.RANDOM:
                    for (int i = 0; i < _rows; i++)
                    {
                        for (int j = 0; j < _cols; j++)
                        {
                            _arr[i *_cols + j] = RandomGenerator.getInstance().Rand(-1, 1);
                        }
                    }
                    break;

            }
        }

        public static Matrix operator + (Matrix p_matrix1, Matrix p_matrix2)
        {
            Matrix res = new Matrix(p_matrix1._rows, p_matrix1._cols);
            Parallel.For(0, p_matrix1._rows * p_matrix1._cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = p_matrix1._arr[i] + p_matrix2._arr[i]; });

            return res;
        }

        public static Matrix operator - (Matrix p_matrix1, Matrix p_matrix2)
        {
            Matrix res = new Matrix(p_matrix1._rows, p_matrix1._cols);
            Parallel.For(0, p_matrix1._rows * p_matrix1._cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = p_matrix1._arr[i] - p_matrix2._arr[i]; });

            return res;
        }

        public static Matrix operator *(Matrix p_matrix1, Matrix p_matrix2)
        {
            Matrix res = new Matrix(p_matrix1._rows, p_matrix2._cols);

            Parallel.For(0, p_matrix1._rows, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => 
            {
                for (int j = 0; j < p_matrix2._cols; j++)
                {
                    for (int k = 0; k < p_matrix1._cols; k++)
                    {
                        res._arr[i * res._cols + j] += p_matrix1._arr[i * p_matrix1._cols + k] + p_matrix2._arr[k * p_matrix2._cols + j];
                    }
                }
            });

            return res;
        }

        public static Vector operator * (Matrix p_matrix1, Vector p_vector1)
        {
            Vector res = new Vector(p_matrix1._rows);

            Parallel.For(0, p_matrix1._rows, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i =>
            {
                for (int j = 0; j < p_matrix1._cols; j++)
                {
                    res[i] += p_matrix1._arr[i * p_matrix1._cols + j] * p_vector1[j];
                }
            });

            return res;
        }

        public static Matrix operator * (Matrix p_matrix1, float p_const)
        {
            Matrix res = new Matrix(p_matrix1._rows, p_matrix1._cols);

            Parallel.For(0, p_matrix1._rows * p_matrix1._cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = p_matrix1._arr[i] * p_const; });

            return res;
        }

        public static Matrix operator *(float p_const, Matrix p_matrix1)
        {
                return p_matrix1 * p_const;
        }

        public Matrix T()
        {
            Matrix res = new Matrix(_cols, _rows);

            Parallel.For(0, _rows, i =>
            {
                for (int j = 0; j < _cols; j++)
                {
                    res._arr[j * res._cols + i] = _arr[i * _cols + j];
                }
            });

            return res;
        }

        public Matrix Inv()
        {
            Matrix res = new Matrix(_rows, _cols);

            Parallel.For(0, _rows * _cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = 1f / _arr[i]; });

            return res;
        }

        public static Matrix Zero(int p_rows, int p_cols)
        {
            return new Matrix(p_rows, p_cols, INIT.ZERO);
        }

        public static Matrix Random(int p_rows, int p_cols)
        {
            return new Matrix(p_rows, p_cols, INIT.RANDOM);
        }

        public static Matrix Identity(int p_rows, int p_cols)
        {
            return new Matrix(p_rows, p_cols, INIT.IDENTITY);
        }

        public Vector Row(int p_index)
        {
            float[] data = new float[_cols];

            for (int i = 0; i < _cols; i++)
            {
                data[i] = _arr[p_index * _cols + i];
            }

            return new Vector(_cols, data);
        }

        public void SetRow(int p_index, Vector p_vector)
        {
            for (int i = 0; i < _cols; i++)
            {
                _arr[p_index * _cols + i] = p_vector[i];
            }
        }

        public Vector Col(int p_index)
        {
            float[] data = new float[_rows];

            for (int i = 0; i < _rows; i++)
            {
                data[i] = _arr[i * _cols + p_index];
            }

            return new Vector(_rows, data);
        }

        public void setCol(int p_index, Vector p_vector)
        {
            for (int i = 0; i < _rows; i++)
            {
                _arr[i * _cols + p_index] = p_vector[i];
            }
        }


        public static Matrix Value(int p_rows, int p_cols, float p_value)
        {
            return new Matrix(p_rows, p_cols, INIT.VALUE, p_value);
        }

        public Matrix Sqrt()
        {
            Matrix res = new Matrix(_rows, _cols);

            Parallel.For(0, _rows * _cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = (float)Math.Sqrt(_arr[i]); });

            return res;
        }

        public Matrix Pow(int p_n)
        {
            Matrix res = new Matrix(_rows, _cols);

            Parallel.For(0, _rows * _cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = (float)Math.Pow(_arr[i], p_n); });

            return res;
        }

        public Matrix Dot(Matrix p_matrix)
        {
            Matrix res = new Matrix(_rows, _cols);

            Parallel.For(0, _rows * _cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = _arr[i] * p_matrix._arr[i]; });

            return res;
        }

        public float this[int p_i, int p_j]
        {

            get
            {
                return _arr[p_i * _cols + p_j];
            }

            set
            {
                _arr[p_i * _cols + p_j] = value;
            }
        }
    }

    public class Vector : Base
    {
        public Vector(int p_dim, float[] p_data) : base(p_dim, 1, p_data)
        {
            
        }

        public Vector(int p_dim, List<float> inputs) : base(p_dim, 1, inputs)
        {

        }

        public Vector(int p_dim = 0, INIT p_init = INIT.ZERO, float p_value = 0) : base(p_dim, 1)
        {
            Init(p_init, p_value);
        }

        public Vector(int p_rows, int p_cols, INIT p_init, float p_value = 0) : base(p_rows, p_cols)
        {
            Init(p_init, p_value);
        }

        public Vector(Vector p_copy) : base(p_copy)
        {
        }

        new void Init(INIT p_init, float p_value)
        {
            switch (p_init)
            {
                case INIT.ZERO:
                    Fill(0);
                    break;
                case INIT.ONES:
                    Fill(1);
                    break;
                case INIT.VALUE:
                    Fill(p_value);
                    break;
                case INIT.RANDOM:
                    for (int i = 0; i < _rows; i++)
                    {
                        _arr[i] = RandomGenerator.getInstance().Rand(-1f, 1f);
                    }
                    break;
            }
        }

        public Vector T()
        {
            Vector res = new Vector(0);

            if (_cols == 1)
            {
                res._rows = 1;
                res._cols = _rows;
                res.Internal_init();
            }
            else if (_rows == 1)
            {
                res._rows = _cols;
                res._cols = 1;
                res.Internal_init();
            }

            for (int i = 0; i < _rows; i++)
            {
                res._arr[i] = _arr[i];
            }

            return res;
        }

        public static Vector operator + (Vector p_vector1, Vector p_vector2)
        {
            Vector res = null;

            if (p_vector1._cols == 1)
            {
                res = new Vector(p_vector1._rows);
            }
            else if (p_vector1._rows == 1)
            {
                res = new Vector(p_vector1._cols);
            }

            Parallel.For(0, res._rows * res._cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = p_vector1._arr[i] + p_vector2._arr[i]; });

            return res;
        }

        public static Vector operator - (Vector p_vector1, Vector p_vector2)
        {
            Vector res = null;

            if (p_vector1._cols == 1)
            {
                res = new Vector(p_vector1._rows);
            }
            else if (p_vector1._rows == 1)
            {
                res = new Vector(p_vector1._cols);
            }

            Parallel.For(0, res._rows * res._cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = p_vector1._arr[i] - p_vector2._arr[i]; });

            return res;
        }

        public static Matrix operator * (Vector p_vector1, Vector p_vector2)
        {
            Matrix res = new Matrix(p_vector1._rows, p_vector2._cols);

            Parallel.For(0, p_vector1._rows, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i =>
            {
                for (int j = 0; j < p_vector2._cols; j++)
                {
                    res[i, j] = p_vector1._arr[i] * p_vector2._arr[j];
                }
            });

            return res;
        }

        public static Vector operator *(float p_const, Vector p_vector1)
        {
            return p_vector1 * p_const;
        }

        public static Vector operator * (Vector p_vector1, float p_const)
        {
            Vector res = null;

            if (p_vector1._cols == 1)
            {
                res = new Vector(p_vector1._rows);

                Parallel.For(0, p_vector1._rows, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = p_vector1._arr[i] * p_const; });
            }
            else if (p_vector1._rows == 1)
            {
                res = new Vector(p_vector1._cols);

                Parallel.For(0, p_vector1._cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { res._arr[i] = p_vector1._arr[i] * p_const; });
            }

            return res;
        }

        public float this[int p_index] {

            get
            {
                return _arr[p_index];
            }

            set
            {
                _arr[p_index] = value;
            }
        }

        public float Norm()
        {
            float res = 0;

            if (_cols == 1)
            {
                for (int i = 0; i < _rows; i++)
                {
                    res += (float)Math.Pow(_arr[i], 2);
                }
            }
            else if (_rows == 1)
            {
                for (int i = 0; i < _cols; i++)
                {
                    res += (float)Math.Pow(_arr[i], 2);
                }
            }

            return (float)Math.Sqrt(res);
        }

        public static Vector Zero(int p_dim)
        {
            return new Vector(p_dim);
        }

        public static Vector Random(int p_dim)
        {            
            return new Vector(p_dim, INIT.RANDOM);
        }

        public static Vector One(int p_dim)
        {
            return new Vector(p_dim, INIT.ONES);
        }

        public static Vector Concat(Vector p_vector1, Vector p_vector2)
        {
            Vector res = new Vector(p_vector1.Size + p_vector2.Size);

            int index = 0;

            for (int i = 0; i < p_vector1.Size; i++)
            {
                res[index] = p_vector1[i];
                index++;
            }

            for (int i = 0; i < p_vector2.Size; i++)
            {
                res[index] = p_vector2[i];
                index++;
            }

            return res;
        }

        public static Vector HadamardProduct(Vector p_vector1, Vector p_vector2)
        {
            float[] d = new float[p_vector1.Size];

            for(int i = 0; i < p_vector1.Size; i++)
            {
                d[i] = p_vector1[i] * p_vector2[i];
            }

            return new Vector(p_vector1.Size, d);
        }

        public int Size
        {
            get { return _rows * _cols; }
        }
    }

    public class Base
    {
        protected int _rows;
        protected int _cols;
        protected float[] _arr;
        public static int PARALLEL = 4;

        protected Base(int p_rows = 0, int p_cols = 0)
        {
            _rows = p_rows;
            _cols = p_cols;

            if (_rows != 0 && _cols != 0)
            {
                Internal_init();
            }
        }

        protected Base(int p_rows, int p_cols, float[] p_data)
        {
            _rows = p_rows;
            _cols = p_cols;

            if (_rows != 0 && _cols != 0)
            {
                Internal_init(p_data);
            }
        }

        protected Base(int p_rows, int p_cols, List<float> p_inputs)
        {
            _rows = p_rows;
            _cols = p_cols;

            if (_rows != 0 && _cols != 0)
            {
                Internal_init(p_inputs);
            }
        }

        protected Base(Base p_copy)
        {
            Clone(p_copy);
        }

        public void Fill(float p_value)
        {
            Parallel.For(0, _rows * _cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL },  i => { _arr[i] = p_value; });
        }

        protected void Init(INIT p_init, float p_value)
        {

        }

        protected void Clone(Base p_copy)
        {
            _rows = p_copy._rows;
            _cols = p_copy._cols;
            _arr = new float[_rows * _cols];

            Parallel.For(0, _rows * _cols, new ParallelOptions { MaxDegreeOfParallelism = PARALLEL }, i => { _arr[i] = p_copy._arr[i]; });
        }

        protected void Internal_init(float[] p_data = null)
        {
            _arr = new float[_rows * _cols];

            if (p_data != null)
            {
                for (int i = 0; i < _rows; i++)
                {
                    for (int j = 0; j < _cols; j++)
                    {
                        _arr[i * _cols + j] = p_data[i * _cols + j];
                    }
                }
            }
        }


        protected void Internal_init(List<float> p_inputs)
        {
            _arr = new float[_rows * _cols];

            int i = 0;
            int j = 0;

            foreach (float v in p_inputs)
            {
                _arr[i * _cols + j] = v;
                j++;
                if (j == _cols)
                {
                    i++;
                    j = 0;
                }
            }

        }

        public float maxCoeff()
        {
            float res = _arr[0];

            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; i++)
                {
                    if (res < _arr[i * _cols + j])
                    {
                        res = _arr[i * _cols + j];
                    };
                }
            }

            return res;
        }

        public float MinCoeff()
        {
            float res = _arr[0];

            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; i++)
                {
                    if (res > _arr[i * _cols + j])
                    {
                        res = _arr[i * _cols + j];
                    };
                }
            }

            return res;
        }

        public int Rows
        {
            get { return _rows; }
        }

        public int Cols
        {
            get { return _cols; }
        }

        public float[] Arr
        {
            get { return _arr; }
        }

        override public string ToString()
        {
            string res = string.Empty;

            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _cols; j++)
                {
                    if (j == _cols - 1)
                    {
                        res += _arr[i * _cols + j] + "\n";
                    }
                    else
                    {
                        res += _arr[i *_cols + j] + ",";
                    }
                }
            }

            return res;
        }
    }
}

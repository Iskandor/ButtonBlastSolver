namespace Sofia
{
    public class AsyncUnit
    {
        private int _asyncSize;
        private int _asyncIndex;
        private bool _active;
        private bool _asyncReady;

        public AsyncUnit()
        {
            _asyncReady = true;
        }

        public void Init(int p_size = 1)
        {
            _asyncSize = p_size;
            _active = p_size > 1;
        }

        public void Reset()
        {
            _asyncIndex = 0;
        }

        public void Update(bool p_terminal)
        {
            if (_active) {
                _asyncReady = false;
                _asyncIndex++;
                if (_asyncIndex == _asyncSize || p_terminal)
                {
                    _asyncReady = true;
                    Reset();
                }
            }
        }

        public bool IsActive
        {
            get { return _active; }
        }

        public bool IsAsyncReady
        {
            get { return _asyncReady; }
        }
    }
}

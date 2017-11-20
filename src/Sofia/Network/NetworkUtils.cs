namespace Sofia
{
    public class NetworkUtils
    {
        public static int KroneckerDelta(int p_i, int p_j)
        {
            return p_i == p_j ? 1 : 0;
        }
    }
}

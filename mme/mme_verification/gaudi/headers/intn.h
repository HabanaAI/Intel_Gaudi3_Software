#ifndef _INT_N_
#define _INT_N_

template< int N>
class intN
{
public:

    intN<N> operator+(intN<N> &b)
    {
        intN<N> retVal;
        for (int i = 0; i < N; i++)
        {
            retVal[i] = arr[i] + b[i];
        }

        return retVal;
    }

    intN<N> operator+(int b)
    {
        intN<N> retVal;
        for (int i = 0; i < N; i++)
        {
            retVal[i] = arr[i] + b;
        }

        return retVal;
    }
    
    intN<N> operator-(intN<N> &b)
    {
        intN<N> retVal;
        for (int i = 0; i < N; i++)
        {
            retVal[i] = arr[i] - b[i];
        }

        return retVal;
    }

    intN<N> operator-(int b)
    {
        intN<N> retVal;
        for (int i = 0; i < N; i++)
        {
            retVal[i] = arr[i] - b;
        }

        return retVal;
    }

    intN<N> operator*(intN<N> &b)
    {
        intN<N> retVal;
        for (int i = 0; i < N; i++)
        {
            retVal[i] = arr[i] * b[i];
        }

        return retVal;
    }

    intN<N> operator*(int b)
    {
        intN<N> retVal;
        for (int i = 0; i < N; i++)
        {
            retVal[i] = arr[i] * b;
        }

        return retVal;
    }

    intN<N> operator/(intN<N> &b)
    {
        intN<N> retVal;
        for (int i = 0; i < N; i++)
        {
            retVal[i] = arr[i] / b[i];
        }

        return retVal;
    }

    intN<N> operator/(int b)
    {
        intN<N> retVal;
        for (int i = 0; i < N; i++)
        {
            retVal[i] = arr[i] / b;
        }

        return retVal;
    }

    intN<N>& operator+=(intN<N> &b)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i]+= b[i];
        }
        return *this;
    }

    intN<N>& operator+=(int b)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] += b;
        }
        return *this;
    }

    intN<N>& operator-=(intN<N> &b)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] -= b[i];
        }
        return *this;
    }

    intN<N>& operator-=(int b)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] -= b;
        }
        return *this;
    }

    intN<N>& operator*=(intN<N> &b)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] *= b[i];
        }
        return *this;
    }

    intN<N>& operator*=(int b)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] *= b;
        }
        return *this;
    }

    intN<N> zero()
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] = 0;
        }
    }

    intN<N>& operator/=(intN<N> &b)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] /= b[i];
        }
        return *this;
    }

    intN<N>& operator/=(int b)
    {
        for (int i = 0; i < N; i++)
        {
            arr[i] /= b;
        }
        return *this;
    }

    int operator[] (int i) { return arr[i]; }
    int* data() { return &(arr[0]); }
private:
    int arr[N];
};

typedef intN<2> int2;
typedef intN<3> int3;
typedef intN<4> int4;
typedef intN<5> int5;
typedef intN<6> int6;
typedef intN<7> int7;
typedef intN<8> int8;


#endif

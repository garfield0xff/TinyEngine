#ifndef __CLIENT_H__
#define __CLIENT_H__

#include <iostream>

using namespace std;

class Client {

public:
    Client(int _c_id = 0, const string& _c_name = " ", int _m_id  = 0);
    string getName() const;
    void setName(string& _c_name);
    void setModel(int m_id);
    int getClientId() const;
    int getModelId() const;
    string getModelName(int m_id) const;

    bool operator==(const Client &other) const;

private: 
    int c_id;
    int m_id;
    // client name
    string c_name;
    //ai model name
    bitset<3> bits;
    

};

#endif // __CLIENT_H
#include "client.h"

Client::Client(int _c_id, const string& _c_name, int _m_id)
    : c_id(_c_id), c_name(_c_name), m_id(_m_id)
{

}

string Client::getName() const {
    return c_name;
}

void Client::setName(string& _c_name){
    this->c_name = _c_name;
}

void Client::setModel(int _m_id) {
    this->bits = bitset<3>(_m_id);
}

int Client::getClientId() const {
    return c_id;
}

int Client::getModelId() const {
    return m_id;
}

string Client::getModelName(int m_id) const {
    string m_name;

    if(m_id == 1) {
        m_name = "CNN";
    }else if (m_id == 2) {
        m_name = "R-CNN";
    }else if (m_id == 3) {
        m_name = "Mobile-Net";
    }else if (m_id == 4) {
        m_name = "YOLO";
    }

    return m_name;
}

bool Client::operator==(const Client &other) const {
    return (this->c_name == other.c_name);
}
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>

#include "client.h"
#include "clientmanager.h"

ClientManager::ClientManager()
{
    ifstream file("clientlist.txt");
    if (!file.fail()) {
        while (!file.eof()) {
            vector<string> row = parseCSV(file, ',');
            if (!row.empty()) {
                int c_id = atoi(row[0].c_str());
                int m_id = atoi(row[2].c_str());
                Client* c = new Client(c_id, row[1], m_id);
                clientList.insert({ c_id, c });
            }
        }
    }
    file.close();
}

ClientManager::~ClientManager()
{
    ofstream file("clientlist.txt");
    if (!file.fail()) {
        for (const auto& v : clientList) {
            Client* c = v.second;
            file << c->getClientId() << "," << c->getName() << "," << c->getModelId() << "\n";
        }
    }
    file.close();
}

void ClientManager::inputClient()
{
    string c_name;
    int m_id;

    cout << "name : "; cin >> c_name;
    cout << "+++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "              Model List                     " << endl;
    cout << "+++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "  1. CNN                                     " << endl;
    cout << "  2. R-CNN                                   " << endl;
    cout << "  3. Mobile-Net                              " << endl;
    cout << "  4. YOLO                                    " << endl;
    cout << "+++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "input number of Model List >> "; cin >> m_id;

    int id = makeId();
    Client* c = new Client(id, c_name, m_id);
    clientList.insert({ id, c });
}

Client* ClientManager::search(int _c_id)
{
    auto it = clientList.find(_c_id);
    if (it != clientList.end()) {
        return it->second;
    }
    return nullptr;
}

void ClientManager::deleteClient(int key)
{
    clientList.erase(key);
}

void ClientManager::modifyClient(int key)
{
    Client* c = search(key);
    if (c) {
        cout << "  ID  |     Name     |     Model Name    |  " << endl;
        cout << setw(5) << setfill('0') << right << c->getClientId() << " | " << left;
        cout << setw(19) << setfill(' ') << right << c->getName() << " | ";
        cout << setw(19) << c->getModelId() << "( " << c->getModelName(c->getModelId()) << " )" <<  " | ";

        string name;
        int m_id;
        cout << "name : "; cin >> name;
        cout << "model number : "; cin >> m_id;

        c->setName(name);
        c->setModel(m_id);
    } else {
        cout << "Client not found." << endl;
    }
}

string printCenterFormat(const string& str, int width)
{
    int len = str.length();
    int padding = (width - len) / 2;
    string result = string(padding, ' ') + str + string(padding, ' ');
    if (result.length() < width) result += ' ';
    return result;
}

void ClientManager::displayInfo()
{
    cout << endl << "  ID  |     CLIENT_NAME     |      MODEL_ID       |" << endl;
    cout << "--------------------------------------------------" << endl;
    for (const auto& v : clientList)
    {
        Client* c = v.second;
        cout << setw(5) << setfill('0') << right << c->getClientId() << " | ";
        cout << setw(19) << setfill(' ') << printCenterFormat(c->getName(), 19) << " | ";
        cout << setw(19) << setfill(' ') << printCenterFormat(to_string(c->getModelId()) + " (" + c->getModelName(c->getModelId()) + ")", 19) << " | " << endl;
    }
    cout << "--------------------------------------------------" << endl;
}




void ClientManager::addClient(Client* c)
{
    clientList.insert({ c->getClientId(), c });
}

int ClientManager::makeId() {
    if (clientList.empty()) {
        return 1;  // ID는 1부터 시작
    } else {
        auto elem = clientList.end();
        int id = (--elem)->first;
        return ++id;
    }
}

vector<string> ClientManager::parseCSV(istream& file, char delimiter)
{
    stringstream ss;
    vector<string> row;
    string t = "\n\r\t";

    while (!file.eof()) {
        char c = file.get();
        if (c == delimiter || c == '\r' || c == '\n') {
            if (file.peek() == '\n') file.get();
            string s = ss.str();
            s.erase(0, s.find_first_not_of(t));
            s.erase(s.find_last_not_of(t) + 1);
            row.push_back(s);
            ss.str("");
            if (c != delimiter) break;
        } else {
            ss << c;
        }
    }

    return row;
}

bool ClientManager::displayMenu()
{
    int ch, key, choice;
    bool running = true;

    while (running)
    {
        cout << "\033[2J\033[1;1H";
        cout << "+++++++++++++++++++++++++++++++++++++++++++++" << endl;
        cout << "              Client Manager                 " << endl;
        cout << "+++++++++++++++++++++++++++++++++++++++++++++" << endl;
        cout << "  1. Display Client List                     " << endl;
        cout << "  2. Input Client                            " << endl;
        cout << "  3. Delete Client                           " << endl;
        cout << "  4. Modify Client                           " << endl;
        cout << "  5. Quit this Program                       " << endl;
        cout << "+++++++++++++++++++++++++++++++++++++++++++++" << endl;
        cout << " What do you wanna do? ";
        cin >> ch;

        switch (ch)
        {
        case 1:
            displayInfo();
            break;
        case 2:
            inputClient();
            break;
        case 3:
            displayInfo();
            cout << "   Choose Key : ";
            cin >> key;
            deleteClient(key);
            break;
        case 4:
            displayInfo();
            cout << "   Choose Key : ";
            cin >> key;
            modifyClient(key);
            break;
        case 5:
            return false;
        }

        cout << endl;
        cout << "1. Return to Menu" << endl;
        cout << "2. Quit Program" << endl;
        cout << "What you wanna do? ";
        cin >> choice;
        if (choice == 2)
        {
            running = false;
        }
    }

    return true;
}

#include <iostream>
#include <client.h>
#include <clientmanager.h>
#include <mlp.h>
#include "engine.h"

int main()
{
    vector<TinyEngine::q8_t> x_data = {1, 2, 3, 4, 5};
    vector<TinyEngine::q8_t> y_data = {2, 4, 6, 8, 10};

    MLP m1(x_data, y_data, 0.01);
    m1.startMLP(500);

    // ClientManager cm;

    // cm.displayMenu();

    return 0;
}
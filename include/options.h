#ifndef __OPTIONS_HEADER__
#define __OPTIONS_HEADER__

#include <iostream>
#include <string>
#include <stdlib.h>
#include <getopt.h>

using namespace std;

class Options {
    string program_name, version_name, version_number;
    string info, help, version, error;

    bool help_mode, version_mode, debug_mode, fast_mode, error_mode;
    bool check_result, shared_memory, padding;

    unsigned int debug_level, fast_level;

    unsigned long m, n, p;
	int sizeX, sizeY;

    void parseOptions(int argc, char *argv[]);
    void parse(int argc, char *argv[]);

    void checkOptions();
    void checkParameters();
public:
    Options(int argc, char *argv[]);
    ~Options();

    void infoPrint();

    string helpPrint();
    string versionPrint();
    string errorPrint();

    string getProgramName();
    string getProgramVersion();
    string getProgramVersionNumber();

    bool helpMode();
    bool versionMode();
    bool errorMode();
    bool debugMode();
    unsigned int getDebugLevel();

    bool checkResult();
    bool usePadding();
    bool sharedMemory();
	int getSizeX();
	int getSizeY();
    unsigned long getM();
    unsigned long getN();
    unsigned long getP();
    bool fastMode();
    unsigned long getFastLevel();
};

#endif /* __OPTIONS_HEADER__*/

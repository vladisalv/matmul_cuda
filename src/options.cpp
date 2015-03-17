#include "options.h"

Options::Options(int argc, char *argv[])
{
    #ifdef PROGRAM_NAME
        program_name = PROGRAM_NAME;
    #else
        program_name = argv[0];
    #endif
    #ifdef VERSION
        version_name = VERSION;
    #else
        version_name = "UNKNOW";
    #endif
    #ifdef VERSION_NUMBER
        version_number = VERSION_NUMBER;
    #else
        version_number = "UNKNOW";
    #endif

    //program_name = version_name = version_number = 0;
    help_mode = version_mode = debug_mode = error_mode = fast_mode = false;
    fast_level = debug_level = 0;

    check_result = shared_memory = padding = false;
	sizeX = sizeY = 0;
    m = n = p = 0;

    help = "help\n";
    version = "version\n";

    parseOptions(argc, argv);
}


Options::~Options()
{
}

void Options::parseOptions(int argc, char *argv[])
{
    parse(argc, argv);
    if (help_mode || version_mode)
        return;
    checkOptions();
    checkParameters();
}

void Options::parse(int argc, char *argv[])
{
    struct option longopts[] = {
        {"help",                    no_argument,       NULL, 'h'},
        {"version",                 no_argument,       NULL, 'v'},
        {"debug-mode",              required_argument, NULL, 'd'},
        {"check",                   no_argument,       NULL, 'c'},
        {"padding",                 no_argument,       NULL, 'k'},
        {"shared",                  no_argument,       NULL, 's'},
        {"sizeX",                   required_argument, NULL, 'x'},
        {"sizeY",                   required_argument, NULL, 'y'},
        {"m",                       required_argument, NULL, 'm'},
        {"n",                       required_argument, NULL, 'n'},
        {"p",                       required_argument, NULL, 'p'},
        {"fast",                    required_argument, NULL, 'f'},
        {0, 0, 0, 0}
    };
    const char *optstring = ":hvd:cksx:y:m:n:p:f:"; // opterr = 0, because ":..."
    int oc;
    int longindex = -1;
    while ((oc = getopt_long(argc, argv, optstring, longopts, &longindex)) != -1) {
        switch (oc) {
        case 'h':
            help_mode = true;
            break;
        case 'v':
            version_mode = true;
            break;
        case 'd':
            debug_mode = true;
            debug_level = atoi(optarg);
            break;
        case 'c':
            check_result = true;
            break;
        case 'k':
            padding = true;
            break;
        case 's':
            shared_memory = true;
            break;
        case 'x':
            sizeX = atoi(optarg);
            break;
        case 'y':
            sizeY = atoi(optarg);
            break;
        case 'm':
            m = atoi(optarg);
            break;
        case 'n':
            n = atoi(optarg);
            break;
        case 'p':
            p = atoi(optarg);
            break;
        case 'f':
			fast_mode = true;
            fast_level = atoi(optarg);
            break;
        case 0: // nothing do
            break;
        case ':':
            error_mode = true;
            error += "unknow option\n";
            break;
        case '?':
        default:
            error_mode = true;
            error += "unknow option\n";
            break;
        }
        longindex = -1;
    }
    //if (optind != argc - 1)
        //error_mode = true;
}

void Options::checkOptions()
{
    if (!m || !n || !p || !sizeX || !sizeY) {
        error_mode = true;
        error += "You forgot direction.\n";
    }
}

void Options::checkParameters()
{
	if (m <= 0 || n <= 0 || p <= 0 || sizeX <= 0 || sizeY <= 0) {
		error_mode = true;
        error += "You forgot direction.\n";
	}
}


string Options::helpPrint()
{
    return help;
}

string Options::versionPrint()
{
    return version;
}

string Options::errorPrint()
{
    return error;
}

void Options::infoPrint()
{
    cout << "m = " << m << endl;
    cout << "n = " << n << endl;
    cout << "p = " << p << endl;
    cout << "sizeX = " << sizeX << endl;
    cout << "sizeY = " << sizeY << endl;
    cout << "check result:  " << (checkResult()  ? "true" : "false") << endl;
    cout << "use padding:   " << (usePadding()   ? "true" : "false") << endl;
    cout << "shared memory: " << (sharedMemory() ? "true" : "false") << endl;
    cout << "fast mode: " << (fastMode() ? "true" : "false");
	if (fastMode())
		cout << fast_level << endl;
	else
		cout << endl;
}

string Options::getProgramName()
{
    return program_name;
}

string Options::getProgramVersion()
{
    return version_name;
}

string Options::getProgramVersionNumber()
{
    return version_number;
}


unsigned int Options::getDebugLevel()
{
    return debug_level;
}

bool Options::helpMode()
{
    return help_mode;
}

bool Options::versionMode()
{
    return version_mode;
}

bool Options::errorMode()
{
    return error_mode;
}

bool Options::debugMode()
{
    return debug_mode;
}





bool Options::checkResult()
{
    return check_result;
}

bool Options::sharedMemory()
{
    return shared_memory;
}

bool Options::usePadding()
{
    return padding;
}

int Options::getSizeX()
{
    return sizeX;
}

int Options::getSizeY()
{
    return sizeY;
}

unsigned long Options::getM()
{
    return m;
}

unsigned long Options::getN()
{
    return n;
}

unsigned long Options::getP()
{
    return p;
}

bool Options::fastMode()
{
    return fast_mode;
}

unsigned long Options::getFastLevel()
{
    return fast_level;
}

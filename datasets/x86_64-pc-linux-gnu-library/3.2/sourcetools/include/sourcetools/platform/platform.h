#ifndef SOURCETOOLS_PLATFORM_PLATFORM_H
#define SOURCETOOLS_PLATFORM_PLATFORM_H

#ifdef _WIN32
# define SOURCETOOLS_PLATFORM_WINDOWS
#endif

#ifdef __APPLE__
# define SOURCETOOLS_PLATFORM_MACOS
#endif

#ifdef __linux__
# define SOURCETOOLS_PLATFORM_LINUX
#endif

#if defined(__sun) && defined(__SVR4)
# define SOURCETOOLS_PLATFORM_SOLARIS
#endif

#endif /* SOURCETOOLS_PLATFORM_PLATFORM_H */

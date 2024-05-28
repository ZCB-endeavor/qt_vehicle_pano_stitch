#ifndef FILESYS_UTIL_H
#define FILESYS_UTIL_H

#include <string>
#include <vector>

namespace filesys_util
{
  // Return the current working directory of the calling application
  bool getCurrentWorkingDirectoryPath(std::string& dir);

  // Returns a list of files in the dir specified
  // If ext is specified only files of the particular extenstion are searched
  bool getFilelistFromDir(const std::string& dir, const std::string& ext, std::vector<std::string>& fileList);
}


#endif

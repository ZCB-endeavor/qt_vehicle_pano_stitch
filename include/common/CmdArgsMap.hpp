/*
* Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#include <string>
#include <string.h>
#include <sstream>
#include <vector>
#include <map>
#include <assert.h>
#include <stdio.h>
#include <typeinfo>

#if defined(_WIN32) || defined(_WIN64)
#define SLASH_DELIM_QUOTED "\\"
#else
#define SLASH_DELIM_QUOTED "/"
#endif

template< typename T1, typename T2 >
struct cmap : public std::map<T1, T2>
{
  cmap() : std::map<T1, T2>() {}
  cmap(const T1& t1, const T2& t2) : std::map<T1, T2>()
  {
    (*this)[t1] = t2;
  }

  inline cmap& operator()(const T1& t1, const T2& t2)
  {
    (*this)[t1] = t2;
    return *this;
  }
};

struct CmdArgList : public std::vector<std::string>
{
  bool m_bPresent;
  std::string m_desc;

  template<typename T>
  inline T t_convert(const char* sz)
  {
    std::cout << "Warning: Command line arguments of type " << typeid(T).name() << " are not supported. Setting the value to zero.\n";
    return T(0);
  }

  template<typename T>
  inline T expect(size_t i)
  {
    if (i < this->size())
    {
      return t_convert<T>((*this)[i].c_str());
    }
    else
    {
      return T(0);
    }
  }

  inline const char* sz(size_t i)
  {
    return i < this->size() ? (*this)[i].c_str() : nullptr;
  }

  inline const std::string& str(size_t i)
  {
    static std::string __emptyString;
    return i < this->size() ? (*this)[i] : __emptyString;
  }
};

  template<>
  inline const char* CmdArgList::t_convert(const char* sz)
  {
    return sz;
  }

  template<>
  inline std::string CmdArgList::t_convert(const char* sz)
  {
    return sz;
  }

  template<>
  inline int CmdArgList::t_convert(const char* sz)
  {
    return atoi(sz);
  }

  template<>
  inline size_t CmdArgList::t_convert(const char* sz)
  {
    return size_t( atoi(sz) );
  }

  template<>
  inline double CmdArgList::t_convert(const char* sz)
  {
    return atof(sz);
  }

  template<>
  inline float CmdArgList::t_convert(const char* sz)
  {
    return float(atof(sz));
  }

class CmdArgsMap : public std::map< std::string, CmdArgList >
{
public:
  CmdArgsMap(const char* token = "-")
  {
    m_maxArgWidth = 0;
    m_token = token;
  }
  CmdArgsMap(int argc, char** argv, const char* token = "-")
  {
    m_maxArgWidth = 0;
    m_token = token;
    parse(argc, argv);
  }
  CmdArgsMap(std::vector< std::string > argList, const char* token = "-")
  {
    m_maxArgWidth = 0;
    m_token = token;
    parse(argList);
  }

  CmdArgsMap& parse(int argc, char** argv)
  {
    size_t tklen = strlen(m_token.c_str());

    int i = 1;
    while (i < argc)
    {
      while ((i < argc) && (strncmp(argv[i], m_token.c_str(), tklen) == 0))
      {
        std::string szCurArg(argv[i] + tklen);
        static_cast<std::map< std::string, CmdArgList >> (*this)[szCurArg] = CmdArgList();
        CmdArgList& curArg = static_cast<std::map< std::string, CmdArgList >&> (*this)[szCurArg];

        i++;
        while ((i < argc) && (strncmp(argv[i], m_token.c_str(), tklen) != 0))
        {
          curArg.push_back(argv[i]);
          i++;
        }
      }

      break;
    }

    if (argc > 1 && this->empty())
    {
      printf("No valid arguments provided. Please use '%s' as a prefix token\n", m_token.c_str());
    }

    return *this;
  }

  CmdArgsMap& parse(std::vector< std::string > argList)
  {
    size_t tklen = strlen(m_token.c_str());

    int i = 0;
    while (i < argList.size())
    {
      while ((i < argList.size()) && (strncmp(argList[i].c_str(), m_token.c_str(), tklen) == 0))
      {
        std::string szCurArg(argList[i].c_str() + tklen);
        static_cast< std::map< std::string, CmdArgList> > (*this)[szCurArg] = CmdArgList();
        static_cast< std::map< std::string, CmdArgList> > (*this)[szCurArg].m_bPresent = true;
        CmdArgList& curArg = static_cast<std::map< std::string, CmdArgList >&> (*this)[szCurArg];

        i++;
        while ((i < argList.size()) && (strncmp(argList[i].c_str(), m_token.c_str(), tklen) != 0))
        {
          curArg.push_back(argList[i].c_str());
          i++;
        }
      }

      break;
    }

    if (argList.size() > 1 && this->empty())
    {
      printf("No valid arguments provided. Please use '%s' as a prefix token\n", m_token.c_str());
    }

    return *this;
  }

  CmdArgList* operator[](const std::string& szIndex)
  {
    std::map< std::string, CmdArgList >::iterator arglist = (*this).find(szIndex);
    return arglist != (*this).end() ? &((*arglist).second) : nullptr;
  }

  template< typename T >
  CmdArgsMap& operator()( const char* szName, const char* szDesc, T* writeTo, const T& defaultVal, bool* bPresent = nullptr )
  {
    CmdArgList& curArgList = static_cast<std::map< std::string, CmdArgList >&> (*this)[szName];
    curArgList.m_desc = szDesc;
    m_maxArgWidth = m_maxArgWidth > strlen(szName) ? m_maxArgWidth : strlen(szName);
    if( !curArgList.sz(0) )
    {
      if( bPresent )
      {
        *bPresent = false;
      }
      *writeTo = defaultVal;
    }
    else
    {
      if (bPresent)
      {
        *bPresent = true;
      }
      *writeTo = curArgList.expect<T>(0);
    }

    return *this;
  }

  CmdArgsMap& operator()(const char* szName, const char* szDesc, bool* bPresent)
  {
    auto baseList = static_cast<std::map< std::string, CmdArgList >&> (*this);
    auto curArgList = baseList.find(szName);

    if( curArgList != baseList.end() )
    {
      m_maxArgWidth = m_maxArgWidth > strlen(szName) ? m_maxArgWidth : strlen(szName);
      (*curArgList).second.m_desc = szDesc;
      (*curArgList).second.m_bPresent = true;
      *bPresent = true;
    }

    return *this;
  }

  CmdArgsMap& operator()(const char* szHelpDesc)
  {
    m_helpDesc = szHelpDesc;
    return *this;
  }

  std::string help()
  {
    std::stringstream ss;

    ss << m_helpDesc << std::endl << std::endl;

    auto argMap = static_cast<std::map< std::string, CmdArgList >> (*this);
    for( auto it = argMap.begin(); it != argMap.end(); it++ )
    {
      size_t curWidth = (*it).first.size();
      ss << m_token << (*it).first
         << std::string(m_maxArgWidth - curWidth + m_token.size(), ' ')
         << (*it).second.m_desc << std::endl;
    }

    return ss.str();
  }

private:
  size_t m_maxArgWidth;
  std::string m_token;
  std::string m_helpDesc;
};

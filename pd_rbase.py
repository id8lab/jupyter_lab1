# C486
# http://vincentarelbundock.github.io/Rdatasets/datasets.html 

import math, datetime
from time import strftime
import rpy2
import rpy2.robjects as RO
from rpy2.robjects.packages import importr
from rpy2.robjects import NA_Real, pandas2ri
from rpy2.rlike.container import TaggedList
import rpy2.rlike.container as rlc
import rpy2.interactive as IR
import rpy2.robjects.packages as Rpacks
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects.packages import PackageData
from rpy2.robjects.conversion import localconverter

from rpy2.robjects import Formula, Environment
import operator


'''
The conversion is automatically happening when calling R functions. For example, when calling the R function base::summary:

base = importr('base')

with localconverter(ro.default_converter + pandas2ri.converter):
  df_summary = base.summary(pd_df)
print(df_summary)


'''


class R(object):
   RR = RO.r
   GB = RO.globalenv
   SV = RO.StrVector
   IV = RO.IntVector
   FV = RO.FloatVector
   DF = RO.DataFrame

   ANON_PACK = '''
      square <- function(x) {
          return(x^2)
      }

      cube <- function(x) {
          return(x^3)
      }
   '''
           
   PACKS = '''
            function() {
                .packages()
            }
           '''
   DOTTED_DATA = '''
            function(name) {
                eval(parse(text=name))
            }
           '''
   SOME_NUMBERS = '''
            function(n, digits) {
                round(runif(n,0,n), digits=digits)
            }
           '''
   PRETTY_FRAME = '''
            function(tab, breaks) {
                cut(tab, breaks=breaks) -> tmp
                as.data.frame(table(tmp))
            }
           '''
   SERROR_SAMP = '''
            function(x) {
                sqrt(var(x, na.rm=T)/(length(x)-sum(is.na(x))))
            }
           '''
   RANDOMSAMPLE = '''
            function(begin, end, size, replace) {
               if (replace){
                  s = sample(begin:end, size=size, replace=TRUE)
               }
               else {
                  s = sample(begin:end, size=end, replace=FALSE)
               }
            }
           '''
   SAMPLE = '''
            function(dset, size) {
               index = sample(1:nrow(dset), size=size)
               dset[index,]
            }
           '''
   COLUMN_EXTRACT = '''
            function(dset, col) {
               as.vector(dset[,col])
            }
           '''
   SAMP_SIZE = '''
            function(x) {
               length(x) - sum(is.na(x))
            }
           '''
   SHOW_ROW = '''
            function(frame, start, end, by) {
               frame[seq(start,end,by)]
            }
           '''
   SLEAF = '''
            function(v, depth=0) {
               library('aplpack')
	       if (depth != 0) {
                 stem.leaf(v, depth=FALSE)
	       }
	       else {
                 stem.leaf(v, depth=TRUE)
	       }
            }
           '''
   STR = '''
            function(dset) {
               return(as.vector(dset))
            }
         '''
   PARETO = '''
            function(table, xlab, ylab) {
               pareto.chart(table, xlab=xlab, ylab=ylab)
            }
           '''
   HIST = '''
            function(x, title, xlabel, colors, bins) {
                 hist(x, right=FALSE, col=colors, main=title, xlab=xlabel,
	                   breaks=bins)
            }
           '''
   LINES = '''
            function(x, y) {
              lines(x,y,type="l")
            }
           '''
   HISTLINES = '''
            function(hist) {
              lines(c(min(hist$breaks),hist$mids,max(hist$breaks)),c(0,hist$counts,0),type="l")
            }
           '''
   PACKDATA = '''
            function(p) {
              data(package = p)
            }
           '''
   AVAIL = '''
            function() {
               data(package = .packages(all.available = TRUE))
            }
           '''
   SLICE = '''
            function(v, i, k) {
                v+i:v+k
            }
           '''
   HELP   = '''

            function(what, pack) {
               help.search(what, package=pack)
            }
           '''
   READTAB   = '''
            function(afile, header) {
               read.table(file=afile,header=header) 
            }
           '''
   CSV   = '''
            function(afile, sep) {
               read.csv(file=afile,head=TRUE,sep=sep) 
            }
           '''
   EXP   = '''
            function(vec1, n) {
               vec1 ^ n
            }
           '''
   DIV   = '''
            function(vec1, vec2) {
               vec1 / vec2
            }
           '''
   MUL   = '''
            function(vec1, vec2) {
               vec1 * vec2
            }
           '''
   ADD   = '''
            function(vec1, vec2) {
               vec1 + vec2
            }
           '''
   COND   = '''
            function(vec1, vec2, cond) {
               vec1[vec2==cond]
            }
           '''
   SUB   = '''
            function(vec1, vec2) {
               vec1 - vec2
            }
           '''
   SPLOT   = '''
            function(x1, y1, ptype, title, x, y) {
	       if (ptype != '') {
                  plot(x1, y1,main=title,type=ptype, xlab=x, ylab=y)
               }
	       else {
                  plot(x1, y1,main='',type='p', xlab=x, ylab=y)
               }
            }
           '''
   LMPLOT   = '''
            function(dset, formula ){
               lmvar = lm(formula, data=dset)
               plot(lmvar)
            }
           '''
   PLOT   = '''
            function(data, ptype, title, x, y) {
	       if (ptype != '') {
                  plot(data,main=title,type=ptype, xlab=x, ylab=y)
               }
	       else {
                  plot(data,main=title,type='p', xlab=x, ylab=y)
               }
            }
           '''
   PACK_CONTENTS   = '''
            function(pack) {
               objects(grep(pack,search()))
            }
           '''
   TAPPLY   = '''
            function(vec1, vec2, func) {
               tapply(vec1, vec2, func)
            }
           '''
   CLOSEST = '''
            function(vec, n) {
                which(abs(vec-n)==min(abs(vec-n))) 
            }
           '''
   JUPYTER_OPT = '''
            function() {
               options(jupyter.plot_mimetypes = 'image/png') 
            }
           '''
   PI   = RR.pi

   objects    = RR['objects']
   ls         = RR['ls']
   search     = RR['search']
   matrix     = RR['matrix']
   example    = RR['example']
   cbind      = RR['cbind']
   rbind      = RR['rbind']
   apply      = RR['apply']
   eapply     = RR['eapply']
   lapply     = RR['lapply']
   sapply     = RR['sapply']
   head       = RR['head']
   sum        = RR['sum']
   sort       = RR['sort']
   rank       = RR['rank']
   order      = RR['order']
   mean       = RR['mean']
   median     = RR['median']
   range      = RR['range']
   round      = RR['round']
   which      = RR['which']
   iqr        = RR['IQR']                # interquartile range
   var        = RR['var']                # variance
   sd         = RR['sd']                 # standard deviation
   sstr       = RR['str']                 # standard deviation
   quantile   = RR['quantile']
   dimnames   = RR['dimnames']
   names      = RR['names']
   reorder    = RR['reorder']
   colsums    = RR['colSums']
   rowsums    = RR['rowSums']
   colmeans   = RR['colMeans']
   rowmeans   = RR['rowMeans']
   rnorm      = RR.rnorm
   dnorm      = RR.dnorm
   pnorm      = RR.pnorm
   qnorm      = RR.qnorm
   layout     = RR.layout
   plot       = RR.plot
   princomp   = RR.princomp       # Principal component analysis
   Rbase       = importr('base')
   Rdatasets   = importr('datasets')
   Rstats      = importr('stats', robject_translations={'format_perc': '_format_perc'})
   Rgraphics   = importr('graphics')
   Rutils      = importr('utils')
   Rlattice    = importr('lattice')
   Ranova      = Rstats.anova
   par         = RR['par']
   source      = RR['source']


   Bmatrix     = Rbase.matrix

   def __init__(self):
       self.pd_active()
       self._dotty = self.func('py_dotted_data', self.DOTTED_DATA)
       self.pdata = self.func('py_packdata', self.PACKDATA)
       self.available = self.func('py_avail',self.AVAIL)()
       self.installed = self.calc('installed.packages')
       self._histlines = self.func('py_draw_histlines', self.HISTLINES)
       self._lines = self.func('py_draw_lines', self.LINES)
       self._hist  = self.func('py_draw_hist', self.HIST)
       self._pareto = self.func('py_draw_pareto', self.PARETO)
       self._sleaf = self.func('py_draw_sleaf', self.SLEAF)
       self._csv   = self.func('py_r_csvread', self.CSV)
       self._tapply = self.func('py_r_csvread', self.TAPPLY)
       self._closest = self.func('py_r_csvread', self.CLOSEST)
       self._jupyter_opt = self.func('py_r_jupyter', self.JUPYTER_OPT)
       self._readtab = self.func('py_r_readtab', self.READTAB)
       self._mul   = self.func('py_r_mul', self.MUL)
       self._exp   = self.func('py_r_exp', self.EXP)
       self._div   = self.func('py_r_div', self.DIV)
       self._add   = self.func('py_r_add', self.ADD)
       self._sub   = self.func('py_r_sub', self.SUB)
       self._str   = self.func('py_r_str', self.STR)
       self._cond   = self.func('py_r_cond', self.COND)
       self._lmplot   = self.func('py_r_lmplot', self.LMPLOT)
       self._plot   = self.func('py_r_plot', self.PLOT)
       self._splot   = self.func('py_r_splot', self.SPLOT)
       self._samps = self.func('py_sample_size', self.SAMP_SIZE)
       self._sample = self.func('py_sample', self.SAMPLE)
       self._randomsample = self.func('py_randomsample', self.RANDOMSAMPLE)
       self._column_ext = self.func('py_column_extract', self.COLUMN_EXTRACT)
       self._serror = self.func('py_serror_samp', self.SERROR_SAMP)
       self._pretty = self.func('py_pretty_frame', self.PRETTY_FRAME )
       self._show_row = self.func('py_show_row', self.SHOW_ROW)
       self._some_nums = self.func('py_show_row', self.SOME_NUMBERS)
       self._packs = self.func('py_packs', self.PACKS)
       self._help = self.func('py_help', self.HELP)
       self._contents = self.func('py_contents', self.PACK_CONTENTS)
       self.anon  = SignatureTranslatedAnonymousPackage(self.ANON_PACK, "anon_pack")
       #self.Rprint = self.GB.get('print')

   def rdata(self, data):
       return RO.r(data)

   def dotty(self, dset):
      return self._dotty(dset)

   def calc(self, what, *args, **kw):
      cc = self.RR[what]
      return cc(*args, **kw)

   def reorder(self, *args):
      cc = self.RR['reorder']
      return cc(*args)

   def dset_list(self):
      return  self.packdata('datasets')

   def par_reset(self):
      return self.Rgraphics.par()

   def packdesc(self, package):
      try:
         return self.pdata(package)
      except:
         return 'not found'

   def packdata(self, package):
      dl =  PackageData(package)
      return dl.names()

   def data_from(self, package, name):
      pp = PackageData(package)
      dset = pp.fetch(name)
      return RO.DataFrame(dset)

   def dotty(self, dset):
      return self._dotty(dset)


   def cube(self, x):
      return self.anon.cube(x)

   def calc(self, what, *args, **kw):
      cc = self.RR[what]
      return cc(*args, **kw)

   def lm0(self, data, formula):
      fmla = Formula(formula)
      return self.calc('lm', formula, data=data)

   def lm(self, vdict, formula):
      fmla = Formula(formula)
      env = fmla.environment
      for k in vdict:
         env[k] = vdict[k]
      return self.calc('lm', formula)

   def methods(self, fname):
      return self.calc('methods', fname)

   def search_for(self, fname):
      return self.calc('getAnywhere', fname)

   def func(self, fname, fdef):
      self.RR( fname + '<- ' + fdef)
      return self.GB[fname]

   def abs(self, vec):
      if type(vec[0]) == int:
         return self.IV(map(abs, vec))
      if type(vec[0]) == float:
         return self.FV(map(abs, vec))

   def dup(self, elem, n):
      if type(elem) == int:
         return self.IV([elem]*n)
      if type(elem) == float:
         return self.FV([elem]*n)
      return self.SV([elem]*n)

   def max(self, vec):
       return self.calc('max', vec)

   def min(self, vec):
       return self.calc('min', vec)

   def attributes(self, dset):
       return self.calc('attributes', dset)

   def rep(self, vec, vec2):
       return self.calc('rep', vec, vec2)

   def join(self, vec, vec2):
       return self.calc('c', vec, vec2)

   def relfreq_table(self, table, rc=0):
       ''' rc = 1 : row percentages. 2 col '''
       if rc == 0:
          return self.calc('prop.table', table)
       return self.calc('prop.table', table, rc)

   def freq_table(self, table, rc=0):
       ''' rc = 1 : row percentages. 2 col '''
       if rc == 0:
          return self.calc('margin.table', table)
       return self.calc('margin.table', table, rc)

   def table(self, tt={}):
       c = self.RR['table']
       return c(**tt)

   def stem_n_leaf(self, table, scale=1.0):
       st = self.RR['stem']
       return st(table, scale=scale)

   def crosstabs(self, dset, tt=()):
       tabd = {}
       for vec in tt:
          v = self._v(dset,vec)
          tabd[vec] = v
       return self.table(tabd)

   def slice(self):
       s = self.func('slice', self.SLICE)
       return s

   def sv(self, svec):
       return self.SV(svec)

   def iv(self, ivec):
       return self.IV(ivec)

   def fv(self, fvec):
       return self.FV(fvec)

   def set_rownames(self, dset, names):
       dset.rownames = names

   def rownames(self, dset):
       return self.calc('rownames', dset, do_NULL=True)

   def pareto(self, tt, xlab='', ylab=''):
       self.library('qcc')
       return self.calc('pareto.chart',tt, xlab,ylab)

   def barfreq(self, tt, vec_type='float'):
       if vec_type == 'int':
          tt = self.iv(tt)
       else:
          tt = self.fv(tt)
       return self.calc('barplot',tt)

   def seq(self, frm, to, by=1):
       return self.calc('seq', frm, to, by)


   def stemleaf(self, vec, depth=False):
       if depth:
          return self._sleaf(vec, depth=0)
       else:
          return self._sleaf(vec, depth=1)

   def hist(self, vec, title='', xlabel='', colors=None, bins=5):
       if colors is None:
          colors = self.SV(['blue','yellow'])
       seq = bins

       print (type(bins))
       if type(bins) != int:
          seq = self.seq(bins[0], bins[1], bins[2])
       return self._hist(vec, title, xlabel, colors, seq)

   def lines(self, x, y):
       return self._lines(x, y)

   def histlines(self, hist):
       return self._histlines(hist)

   def package_desc(self, pack):
       return self.calc('packageDescription', pack)

   def frame_struct(self, dset):
       return self.calc('str', dset)

   def table_frame(self, table):
       return self.calc('as.data.frame', table)

   def dataframe(self, dset=[]):  
       ''' dataf.rx(1) selects column 1 of dataf 
           dset == list of tuples:('label', vector)'''
       od = rlc.OrdDict(dset)
       return  self.DF(od)

   def dataset(self, dset, brackets=False):
       ''' from package 'datasets'  '''

       if brackets:
          return self._str(self.RR[dset])
       return  self.Rdatasets.__rdata__.fetch(dset)[dset] 

   def factor(self, vec):
       return  RO.FactorVector(vec)

   def is_na(self, vec):
       return self.calc('is.na',vec)

   def cut(self, *args, **kw):
       labels = kw.pop('labels', None)
       if labels is None:
          return self.calc('cut', *args)
       else:
          return self.calc('cut', *args, labels=labels)

   def pretty(self, *args):
       return self.calc('pretty', *args)

   def search(self):
       return self.calc('search')

   def detach(self, pack):
       return self.calc('detach', 'package:'+pack)

   def library(self, pack):
       return self.calc('library', pack)

   def count(self, iv, k):
       return list(iv).count(k)

   def random(self, range, size, replace=True):
       return self._randomsample(range[0], range[1], size, replace)

   def sample(self, dset, size):
       return self._sample(dset, size)

   def column_ext(self, dset, col):
       return self._column_ext(dset, col)

   def prob_sample(self, vec, size):
       return self.calc('sample', vec, size, replace=True)

   def samp_size(self, vec):
       return self._samps(vec)

   def samp_error(self, vec):
       ''' standard error of sample mean '''
       return self._serror(vec)

   def pretty_fr(self, vec, breaks=10):
       ''' standard error of sample mean '''
       return self._pretty(vec, breaks=breaks)

   def _v(self, dset, name):
       try:
           index = dset.names.index(name)
       except:
           return name + ' not found'
       return  dset[index]

   def summary(self, dset):
       return self.calc('summary',dset)

   def translate(self):
       pack = importr('base')
       return pack.scan._prm_translate

   def pretty_letters(self):
       _letters = self.RR['letters']
       rcode = 'paste(%s, collapse="-")' %(_letters.r_repr())
       return self.RR(rcode)

   def letters(self):
       _letters = self.RR['letters']
       return _letters

   def read_tab(self, afile, header=True):
       return self._readtab(afile, header=header)

   def pclosest(self, vec, x):
       xvec = self.dup(x,len(vec))
       a = self.abs(self.sub(vec,xvec))
       mina = min(a)
       for i in range(len(a)):
          if a[i] == mina:
             return vec[i]
       return 0

   def closest(self, vec, n):
       ind =  self._closest(vec, n)
       return vec[ind[0]-1]

   def jupyter_opt(self):
       return self._jupyter_opt()


   def tapply(self, vec1, vec2, func):
       return self._tapply(vec1, vec2, func)

   def read_csv(self, afile, sep=','):
       return self._csv(afile, sep=sep)

   def exp(self, vec1, vec2):
       return self._exp(vec1, vec2)

   def mul(self, vec1, vec2):
       return self._mul(vec1, vec2)

   def div(self, vec1, vec2):
       return self._div(vec1, vec2)

   def add(self, vec1, vec2):
       return self._add(vec1, vec2)

   def sub(self, vec1, vec2):
       return self._sub(vec1, vec2)

   def fit(self, vdict, formula):
       fmla = Formula(formula)
       env = fmla.environment
       for k in vdict:
         env[k] = vdict[k]
       return self.Rstats.lm(fmla)

   def cond(self, vec1, vec2, cond):
       return self._cond(vec1, vec2, cond)

   def splot(self, x1, y1, ptype='', label=('','','') ):
       return self._splot(x1, y1, ptype
                             ,label[0], label[1], label[2])

   def lmplot(self, formula, data):
       fmla = Formula(formula)
       return self._lmplot(data, formula)

   def myplot(self, data, ptype='', label=('','','') ):
       return self._plot(data, ptype
                             ,label[0], label[1], label[2])

   def show_row(self, frame, start, end, by):
       return self._show_row(frame, start, end, by)

   def help(self, what, package):
       print( self._help(what,package))

   def some_nums(self, n, round=1):
       return self._some_nums(n,round)

   def packages(self):
       return self._packs()

   def where(self, obj):
      env = Rpacks.wherefrom(obj)
      try:
         return env.do_slot('name')[0]
      except:
         return 'Not Found'

   def bmatrix(self, nrow=100, ncol=10):
       # create a numerical matrix of size 100x10 filled with NAs
       m = self.Bmatrix(NA_Real, nrow=nrow, ncol=ncol)

       # fill the matrix
       for row_i in xrange(1, nrow+1):
           for col_i in xrange(1, ncol+1):
               m.rx[TaggedList((row_i, col_i))] = row_i + col_i * 100
       return m

   def pddate2str(self, date, fmt):
      return strftime(date, fmt)

   def R2pd(self, rdf):
       with localconverter(RO.default_converter + pandas2ri.converter):
           pd_from_r_df = RO.conversion.rpy2py(rdf)

       return pd_from_r_df


   def pd_active(self):
      pandas2ri.activate()



R = R()




class MCarlo(object):

   def generate(self, size, nrange):
      return R.prob_sample(R.SV(range(nrange[0], nrange[1]+1)), size)

   def count(self, vec, elem):
      return list(vec).count(elem)

   def repeat_exp(self, num_exp, size, nrange, elem):
      l  = []
      for i in range(num_exp):
          t = self.generate(size, nrange)
          c = self.count(t, elem)
          l.append(c)
      return l

   def count_g(self, llist, elem):
      c = 0
      for item in llist:
         if item > elem:
            c += 1
      return c


MC = MCarlo()








if __name__ == '__main__':

  latt = R.packdesc('lattice') 
  mort = R.data_from('lattice', 'USMortality')
  pd_mort = R.R2pd(R.data_from('lattice', 'USMortality'))
  mtcars = R.dataset('mtcars')    # does not need R2pd
  #print(mtcars.head())
  R.packages()
  R.library('lattice') # loads package
  #print(R._v(mtcars, 'cyl'))
  #print(type(mort))

  print(type(R.dotty('state.division')))

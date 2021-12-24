let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd ~/projects/mst/dl/radar
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
$argadd lightning.py
tabnew
tabrewind
edit main.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 126 + 127) / 254)
exe 'vert 2resize ' . ((&columns * 127 + 127) / 254)
argglobal
balt ~/projects/mst/dl/radar/config.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=99
setlocal fml=1
setlocal fdn=20
setlocal fen
37
normal! zo
let s:l = 36 - ((31 * winheight(0) + 25) / 51)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 36
normal! 04|
wincmd w
argglobal
if bufexists("~/projects/mst/dl/radar/config.py") | buffer ~/projects/mst/dl/radar/config.py | else | edit ~/projects/mst/dl/radar/config.py | endif
if &buftype ==# 'terminal'
  silent file ~/projects/mst/dl/radar/config.py
endif
balt ~/projects/mst/dl/radar/models.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=99
setlocal fml=1
setlocal fdn=20
setlocal fen
11
normal! zo
12
normal! zo
24
normal! zo
40
normal! zo
let s:l = 44 - ((10 * winheight(0) + 25) / 51)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 44
normal! 024|
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 126 + 127) / 254)
exe 'vert 2resize ' . ((&columns * 127 + 127) / 254)
tabnext
edit data_utils.py
argglobal
balt lightning_models.py
setlocal fdm=indent
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=99
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 53 - ((25 * winheight(0) + 25) / 51)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 53
normal! 050|
tabnext 1
badd +15 kfold.py
badd +158 lightning_models.py
badd +41 data_utils.py
badd +55 ~/projects/mst/dl/radar/models.py
badd +46 create_submission.py
badd +27 ~/projects/mst/dl/radar/config.py
badd +38 main.py
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOFcI
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :

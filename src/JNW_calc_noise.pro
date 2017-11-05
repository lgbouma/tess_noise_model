; Josh Winn's IDL noise calculator, downloaded from the TESS wiki.

pro calc_noise, $
;
; mandatory inputs
;
   imag, $                          ; apparent mag in Cousins I band
   exptime, $                       ; total exposure time in seconds
;
; mandatory outputs
;
   noise, $                         ; fractional rms (= 1/SNR)
;
; optional inputs
;
   teff=teff, $                     ; effective temperature in Kelvins
   elon=elon, elat=elat, $          ; ecliptic coordinates in degrees
   subexptime=subexptime, $         ; subexposure time (n_exp = exptime/subexptime)
   npix_aper = npix_aper, $         ; number of pixels in photometric aperture
   frac_aper = frac_aper, $         ; fraction of flux enclosed in photometric aperture
   e_pix_ro = e_pix_ro, $           ; rms in no. photons/pixel from readout noise
   geom_area = geom_area, $         ; geometric collecting area
   pix_scale = pix_scale, $         ; arcsec per pixel
   sys_limit = sys_limit, $         ; minimum uncertainty in 1 hr of data, in ppm
   verbose=verbose, $               ; request verbose output
   red=red, $		     	    ; consider a redder bandpass
   al_bk = al_bk, $		    ; Use Al's sky background
   al_phot = al_phot, $             ; Use Al's photomery
   field_angle=field_angle, $	    ; Field angle for effective area
   fov_ind=fov_ind, $ 		    ; FOV index for bk model (0-3)
   bk_p=bk_p, $		    	    ; Background polynomial fit
   frac_off=frac_off, $	 	    ; Fractional offset file (mag vs. pix) 
   bin_sys=bin_sys, $		    ; Is this a binary?
   bin_sep=bin_sep, $		    ; Separation to binary
   bin_imag=bin_imag, $		    ; imag of binary
   bin_teff=bin_teff, $		    ; teff of binary

; optional outputs
;
   noise_star=noise_star, $         ; noise from star counts alone
   noise_sky=noise_sky, $           ; noise from sky counts only
   noise_ro=noise_ro,$              ; noise from readout only
   noise_sys=noise_sys, $           ; noise from systematic limit only
   noise_bin=noise_bin, $	    ; noise from binary companion
   e_star_sub=e_star_sub, $ 	    ; subexposure electron count (for saturation check)
   
   dilution=dilution		    ; background dilution factor
;
;
  if (keyword_set(teff)) then teff=teff else teff=5000.
  if (keyword_set(elon)) then elon=elon else elon=0.0
  if (keyword_set(elat)) then elat=elat else elat=30.0
  if (keyword_set(subexptime)) then subexptime=subexptime else subexptime=2.0
  if (keyword_set(e_pix_ro)) then e_pix_ro=e_pix_ro else e_pix_ro=10.
  if (keyword_set(geom_area)) then geom_area=geom_area else geom_area=73.0
  if (keyword_set(pix_scale)) then pix_scale=pix_scale else pix_scale=20.0
  if (keyword_set(sys_limit)) then sys_limit=sys_limit else sys_limit=60.0
  if (keyword_set(verbose)) then v=1 else v=0
  if (keyword_set(red)) then red=red else red=0
  if (keyword_set(al_phot)) then al_phot=al_phot else al_phot=0
  if (keyword_set(al_bk)) then al_bk=al_bk else al_bk=0
  if (keyword_set(field_angle)) then field_angle=field_angle else field_angle=0.0
  if (keyword_set(al_phot)) then field_angle=0.0
  

  if (keyword_set(npix_aper)) then begin
     npix_aper=npix_aper
  endif else begin
     npix_aper=optimal_npix(imag)
  endelse

  if (keyword_set(frac_aper)) then begin
     frac_aper=frac_aper[fov_ind]
  endif else begin
     ;frac_aper = 0.76
     fwhm = 1.2
     r_aper = sqrt(npix_aper/!PI)
     frac_aper = 1.0 - (fwhm/(2.*!PI)) / ((fwhm/2.)^2 + r_aper^2.) ; from Roland 9 July 2013
  endelse
        
  omega_pix = pix_scale^2.
  n_exposures = exptime/subexptime

  if(v) then begin
     print, 'imag = ', imag
     print, 'exptime = ', exptime
     print, 'teff = ', teff
     print, 'elon = ', elon
     print, 'elat = ', elat
     print, 'npix_aper = ', npix_aper
     print, 'frac_aper = ', frac_aper
     print, 'subexptime = ', subexptime
     print, 'n_exposures = ', n_exposures
     print, 'e_pix_ro = ', e_pix_ro
     print, 'geom_area = ', geom_area
     print, 'pix_scale = ', pix_scale
     print, 'omega_pix = ', omega_pix
     print, 'sys_limit = ', sys_limit
  endif

; electrons from the star

  if (red) then  megaph_s_cm2_0mag = 1.3804467 - 0.06911869*(teff-5000.)/5000. $
  else if (al_phot) then megaph_s_cm2_0mag = 1.7129 + 0.5185*(teff-5000.)/5000. + 2.4616*((teff-5000.)/5000.)^2. $
  ;else megaph_s_cm2_0mag = 1.6301336 + 0.14733937*(teff-5000.)/5000.
  else begin 
; 75 um depletion
;	megaph_s_cm2_0mag = 1.5838 + 0.2878*(teff-3500.)/3500. - 0.1398*((teff-3500.)/3500.)^2.
;       cool = where(teff lt 3500.)
;       if (cool[0] ne -1) then megaph_s_cm2_0mag[cool] = 1.5838
; 100 um depletion
       megaph_s_cm2_0mag = 1.6685 + 0.2145*(teff-3500.)/3500. - 0.0945*((teff-3500.)/3500.)^2.
       cool = where(teff lt 3500.)
       if (cool[0] ne -1) then megaph_s_cm2_0mag[cool] = 1.6685
  endelse
  e_star = 10.0^(-0.4*imag) * 1D6 * megaph_s_cm2_0mag * $
	geom_area * cos(!DPI * field_angle/180.)* exptime * frac_aper
  e_star_sub = e_star*subexptime/exptime

  if (v) then print, 'e_star = ', e_star

; e/pix from zodi

  dlat = (abs(elat)-90.)/90.
  if (al_bk) then vmag_zodi=22. $
  else vmag_zodi = 23.345 - 1.148*dlat^2.
  e_pix_zodi = 10.0^(-0.4*(vmag_zodi-22.8)) * 2.56D-3 * $
	geom_area * cos(!DPI * field_angle/180.) * omega_pix * exptime
  if (red) then e_pix_zodi = e_pix_zodi*0.828

  if (v) then print, 'vmag_zodi = ', vmag_zodi
  if (v) then print, 'e_pix_zodi = ', e_pix_zodi

; e/pix from background stars

  euler, elon, elat, glon, glat, select=5
  if (v) then print, 'glon, glat = ', glon, glat

  dlon = glon
  q = where(dlon gt 180.) & if(q[0] ne -1) then dlon[q] = 360.-dlon[q]
  dlon = abs(dlon)/180.0D0
  if keyword_set(bk_p) then begin
    ;print, 'Using new model'
    p = bk_p[fov_ind,*]
    dlat = abs(glat)/90.0D0
    imag_bgstars = p[*,0] + $
	p[*,1]*(1.0-exp(-dlon/p[*,2])) + $
	p[*,3]*(1.0-exp(-dlat/p[*,4])) + $
	p[*,5]*sqrt(dlon*dlat)
  endif else begin 
    dlat = abs(glat)/40.0D0
    p = [18.9733D0, 8.833D0, 4.007D0, 0.805D0]
    imag_bgstars = p[0] + p[1]*dlat + p[2]*dlon^(p[3])
  endelse

  e_pix_bgstars = 10.0^(-0.4*imag_bgstars) * 1.7D6 * $
	geom_area * cos(!DPI * field_angle/180.) * omega_pix * exptime
  if (red) then e_pix_bgstars = e_pix_bgstars*0.828
  if (al_bk) then e_pix_bgstars = 0.0

  if (v) then print, 'imag_bgstars = ', imag_bgstars
  if (v) then print, 'e_pix_bgstars = ', e_pix_bgstars

; compute noise sources

  e_tot_sky = npix_aper*(e_pix_zodi + e_pix_bgstars) 

  e_bin = dblarr(n_elements(e_tot_sky))
; e/pix from companion
  if (total(bin_sys) gt 0) then begin
    bins = where(bin_sys)
    pix_sep = bin_sep[bins]/pix_scale  	; from arcsec to pixels
    r = frac_off[fov_ind[bins],*]	; distance (in pixels)
    di = frac_off[fov_ind[bins]+4,*]    ; imag attenuation
    dibin = interpol(di, r, pix_sep)    ; interpolate over spline fit
    bin_imag_att = bin_imag[bins] + dibin 
    
    bin_teffs = bin_teff[bins]
    megaph_s_cm2_0mag = 1.6685 + 0.2145*(bin_teffs-3500.)/3500. - $
	0.0945*((bin_teffs-3500.)/3500.)^2.
    cool = where(bin_teffs lt 3500.)
    if (cool[0] ne -1) then megaph_s_cm2_0mag[cool] = 1.6685
    e_bin[bins] = 10.0^(-0.4*bin_imag_att) * 1D6 * megaph_s_cm2_0mag * $
	geom_area * cos(!DPI * field_angle/180.)* exptime * frac_aper
  endif
  
  e_tot = e_tot_sky + e_star + e_bin
  
  noise_star = sqrt(e_star) / e_tot
  noise_sky  = sqrt(e_tot_sky) / e_tot
  noise_ro   = sqrt(npix_aper*n_exposures)*e_pix_ro / e_tot
  noise_bin  = sqrt(e_bin) / e_tot
  noise_sys  = 0.0*noise_star + sys_limit/1d6/sqrt(exptime/3600.)

  dilution = e_tot / e_star
  ;if (al) then noise = sqrt(1.0/(e_star + e_tot_sky) + noise_sys^2.) else $
  noise = sqrt( noise_star^2. + noise_sky^2. + noise_ro^2. + noise_sys^2. + noise_bin^2. )
  if (v) then begin
     print, 'noise_star [ppm] = ', noise_star*1d6
     print, 'noise_sky  [ppm] = ', noise_sky*1d6
     print, 'noise_ro   [ppm] = ', noise_ro*1d6
     print, 'noise_sys  [ppm] = ', noise_sys*1d6
     print, 'noise      [ppm] = ', noise*1d6
  endif

end
